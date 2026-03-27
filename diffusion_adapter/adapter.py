"""
diffusion_adapter.py — Main entry point for DiffusionPlanner integration.
 
Exposes a single function:
    diffusion_plan(cur, destination, obs) -> List[TrajectoryPoint]
 
which is called by Car._call_planner() in v2_diffusion.py.
 
On first call the model is loaded from disk. Subsequent calls reuse the
loaded model (module-level singleton).
 
Coordinate conventions
----------------------
- CARLA uses left-handed coords (Y left, yaw CW in degrees).
- DiffusionPlanner uses right-handed ego-centric coords.
- coord_utils.py handles all conversions.
"""

import os
import torch
import numpy as np
from typing import List, Optional
import sys

### HACK:
_DP_PKG_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Diffusion-Planner')                                                            
if _DP_PKG_ROOT not in sys.path:                                                                                                                        
    sys.path.insert(0, _DP_PKG_ROOT)                                                                                                                    
for _k in [k for k in sys.modules if k.startswith('diffusion_planner.utils') or                                                                         
            k.startswith('diffusion_planner.model')]:                                                                                                    
    del sys.modules[_k]  
 

# sys.path.insert(0, DIFFUSION_REPO_PATH)
 
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from diffusion_planner.model.guidance.collision import collision_guidance_fn
from diffusion_planner.data_process.utils import (
    convert_to_model_inputs,
    _state_se2_array_to_transform_matrix,
    _state_se2_array_to_transform_matrix_batch,
    _transform_matrix_to_state_se2_array_batch,
)

from utils.coord_utils import (
    carla_transform_to_standard,
    carla_velocity_to_standard,
    standard_to_carla,
)
from utils.agent_process import (
    AgentHistoryBuffer,
    AgentState,
    build_neighbor_agents_past,
    build_static_objects,
)
from v2 import TrajectoryPoint, Direction, MIN_SPEED, MAX_SPEED, plan_hybrid_a_star
from utils.map_process import build_map_features, set_astar_path, LANE_NUM, ROUTE_LANE_NUM, get_dist


GOAL_GUIDANCE_SCALE = 1.0
PATH_GUIDANCE_SCALE = 10.0


class _CombinedGuidance:
    """
    Guidance function following the GuidanceWrapper pattern:
    1. One-step denoising correction for better x0 estimate
    2. Denormalize state and observations
    3. Apply collision + path-following + goal energy functions
    4. Only active at late diffusion steps (t < 0.1)
    """
    def __init__(self):
        self._goal = None       # (2,) tensor: destination in ego-centric standard frame
        self._path = None       # (M, 2) tensor: A* path in ego-centric standard frame

    def set_goal(self, gx: float, gy: float):
        self._goal = torch.tensor([gx, gy], dtype=torch.float32).to(_device)

    def set_path(self, path_ego: np.ndarray):
        """Set A* path points in ego-centric standard frame. (M, 2)"""
        self._path = torch.tensor(path_ego, dtype=torch.float32).to(_device)

    def __call__(self, x, t, cond, inputs, *args, **kwargs):
        inputs = kwargs.get('inputs')
        state_normalizer = kwargs.get('state_normalizer')
        observation_normalizer = kwargs.get('observation_normalizer')
        model = kwargs.get('model')
        model_condition = kwargs.get('model_condition')

        B, P, _ = x.shape

        # --- One-step denoising correction (from GuidanceWrapper) ---
        with torch.no_grad():
            x_fix = model(x, t, **model_condition) - x.detach()
        x_fix = x_fix.reshape(B, P, -1, 4)
        x_fix[:, :, 0] = 0.0  # don't correct the current state
        x = x + x_fix.reshape(B, P, -1)

        # --- Denormalize state and observations ---
        x_real = state_normalizer.inverse(x.reshape(B, P, -1, 4))
        inputs_real = observation_normalizer.inverse(inputs)

        # --- Collision guidance (operates on denormalized data) ---
        collision_energy = collision_guidance_fn(x_real, t, cond, inputs_real, *args, **kwargs)

        # --- Only apply path/goal guidance at late diffusion steps ---
        if not (t < 0.1 and t > 0.005):
            return collision_energy

        energy = collision_energy

        # --- Path-following guidance: penalize distance from A* at each timestep ---
        if self._path is not None and len(self._path) >= 2:
            ego_traj = x_real[:, 0, 1:, :2]  # [B, T, 2] — skip current state
            # Subsample path to limit memory (max 200 points)
            path = self._path
            if len(path) > 200:
                indices = torch.linspace(0, len(path) - 1, 200).long()
                path = path[indices]
            # Distance from each trajectory point to nearest A* path point
            diff = ego_traj[:, :, None, :] - path[None, None, :, :]  # [B, T, M, 2]
            dist_to_path = torch.norm(diff, dim=-1)  # [B, T, M]
            min_dist, _ = dist_to_path.min(dim=-1)   # [B, T]
            path_energy = -PATH_GUIDANCE_SCALE * min_dist.mean()
            energy = energy + path_energy

        # --- Goal guidance: pull final point toward destination ---
        if self._goal is not None:
            ego_final = x_real[:, 0, -1, :2]  # [B, 2]
            dist = torch.norm(ego_final - self._goal[None], dim=-1)  # [B]
            goal_energy = -GOAL_GUIDANCE_SCALE * dist.mean()
            energy = energy + goal_energy

        return energy


_combined_guidance = _CombinedGuidance()


### Model variables

_model: Optional[Diffusion_Planner] = None
_config: Optional[Config] = None
_history_buffer: AgentHistoryBuffer = AgentHistoryBuffer()
_device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
_astar_path_set: bool = False
 
# Normalizer — loaded from config
_observation_normalizer = None


def load_model(ckpt_path, args_path):
    ## load model and cache

    global _model, _config, _observation_normalizer

    
    _config = Config(args_path, guidance_fn=None)
    _observation_normalizer = _config.observation_normalizer

    _model = Diffusion_Planner(_config)
    state_dict = torch.load(ckpt_path, map_location=_device)

    ###

    if 'ema_state_dict' in state_dict:
        state_dict = state_dict['ema_state_dict']
    elif 'model' in state_dict:
        state_dict = state_dict['model']


    state_dict = {
        (k[len('module.'):] if k.startswith('module.') else k): v
        for k, v in state_dict.items()
    }

    _model.load_state_dict(state_dict)
    _model.eval()
    _model = _model.to(_device)

    print(f"[diffusion_adapter] Model loaded on {_device}")



def _build_model_inputs(cur: 'TrajectoryPoint', obs) -> dict:
    """
    Convert the current CARLA scene into the dict of tensors DiffusionPlanner expects.
 
    :param cur: current ego TrajectoryPoint (CARLA frame)
    :param destination: destination TrajectoryPoint (CARLA frame)
    :param obs: ObstacleMap (used to get dynamic agent states)
    :return: dict of tensors ready for the model
    """
    ego_x, ego_y, ego_heading = carla_transform_to_standard(cur.x, cur.y, np.rad2deg(cur.angle))

    anchor_ego_state = np.array([ego_x, ego_y, ego_heading], dtype=np.float64)
    agent_states = []
    
    if hasattr(obs, 'dyn_obs_clusters'):
        for actor_id, cluster in obs.dyn_obs_clusters.items():
            mean = cluster['state_mean']  # [x, y, vx, vy] in CARLA global frame
            bb   = cluster['bb']
            # Derive heading from velocity; fall back to actual CARLA yaw if stationary
            vx_carla, vy_carla = mean[2], mean[3]
            if abs(vx_carla) > 0.01 or abs(vy_carla) > 0.01:
                yaw = np.arctan2(vy_carla, vx_carla)
            else:
                yaw = cluster.get('yaw', 0.0)  # use true orientation for parked cars

            # Convert to standard frame
            std_x, std_y, std_heading = carla_transform_to_standard(
                mean[0], mean[1], np.rad2deg(yaw)
            )
            _, _, std_heading = carla_transform_to_standard(0, 0, np.rad2deg(cur.angle))
            std_vx = cur.speed * np.cos(std_heading)
            std_vy = cur.speed * np.sin(std_heading)

            agent_states.append(AgentState(
                actor_id = actor_id,
                x        = std_x,
                y        = std_y,
                heading  = std_heading,
                vx       = std_vx,
                vy       = std_vy,
                width    = float(bb.extent.y * 2),
                length   = float(bb.extent.x * 2),
                agent_type = cluster.get('actor_type', 'vehicle'),
            ))


    ## history logic for all external agents
    _history_buffer.update(agent_states)
    # neighbor_agents_past = build_neighbor_agents_past(_history_buffer, anchor_ego_state)
    neighbor_agents_past = np.zeros((32, 21, 11), dtype=np.float32)

    # --- Static objects (none from ObstacleMap, leave zeros) ---
    # static_objects = build_static_objects()
    static_objects = np.zeros((5, 10), dtype=np.float32)

    lanes, route_lanes = build_map_features(anchor_ego_state)

    print("Route lane 0 points (ego frame x,y):", route_lanes[0, :5, :2])

    dist_along = get_dist(anchor_ego_state)
    std_vx, std_vy = carla_velocity_to_standard(cur.speed * np.cos(cur.angle), cur.speed * np.sin(cur.angle))

    ### what is the distance along the A* path that has already been traveled
    ego_current_state = np.array([
        # position in ego frame (this is always 0)
        0., 0., 1., 0.,   # sin heading
        std_vx, std_vy,      # v
        0., 0.,      # ax, ay
        0., 0.,      # steering angle, yaw rate
    ], dtype=np.float32)

    data = {
        'neighbor_agents_past': neighbor_agents_past[:, -21:],  # last 21 steps
        'ego_current_state':    ego_current_state,
        'static_objects':       static_objects,
        'lanes':                lanes,
        'lanes_speed_limit':           np.full((LANE_NUM, 1), MAX_SPEED, dtype=np.float32),
        'lanes_has_speed_limit':       np.ones((LANE_NUM, 1), dtype=np.bool_),
        'route_lanes':                 route_lanes,
        'route_lanes_speed_limit':     np.full((ROUTE_LANE_NUM, 1), MAX_SPEED, dtype=np.float32),
        'route_lanes_has_speed_limit': np.ones((ROUTE_LANE_NUM, 1), dtype=np.bool_),

    }

    return convert_to_model_inputs(data, _device)

def _parse_model_output(outputs, anchor_ego_state, cur):
    predictions = outputs['prediction'][0, 0].detach().cpu().numpy().astype(np.float64)
    T = predictions.shape[0]

    # print(f"[diffusion_adapter] Raw ego-centric predictions (first 5): {predictions[:5, :2]}")

    headings = np.arctan2(predictions[:, 3], predictions[:, 2])  # (T,)
    local_poses = np.stack([predictions[:, 0], predictions[:, 1], headings], axis=1)  # (T, 3)

    anchor_matrix = _state_se2_array_to_transform_matrix(anchor_ego_state)
    pose_matrices = _state_se2_array_to_transform_matrix_batch(local_poses)
    global_matrices = anchor_matrix @ pose_matrices
    global_poses = _transform_matrix_to_state_se2_array_batch(global_matrices)

    trajectory = []
    for i in range(T):
        gx, gy, gh = global_poses[i]
        carla_x, carla_y, carla_yaw_deg = standard_to_carla(gx, gy, gh)
        carla_heading_rad = np.deg2rad(carla_yaw_deg)

        if i == 0:
            speed = MIN_SPEED
        else:
            prev_gx, prev_gy = global_poses[i-1, :2]
            dt = 0.1  # 10Hz
            speed = max(MIN_SPEED, np.sqrt((gx-prev_gx)**2 + (gy-prev_gy)**2) / dt)
 
        tp = TrajectoryPoint(Direction.FORWARD, carla_x, carla_y, speed, carla_heading_rad)
        trajectory.append(tp)

    print(f"Anchor ego state (standard): {anchor_ego_state}")
    print(f"Raw predictions ego frame (first 3): {predictions[:3, :2]}")
    print(f"Global poses (first 3): {global_poses[:3, :2]}")
    print(f"CARLA output (first 3): {[(t.x, t.y) for t in trajectory[:3]]}")
    print(f"Cur CARLA pos: {cur.x}, {cur.y}")

    h = anchor_ego_state[2]
    px, py = predictions[0, :2]
    manual_x = anchor_ego_state[0] + px*np.cos(h) - py*np.sin(h)
    manual_y = anchor_ego_state[1] + px*np.sin(h) + py*np.cos(h)
    print(f"Manual transform: ({manual_x}, {manual_y})")
    print(f"SE2 transform: {global_poses[0, :2]}")
    
    return trajectory
    
def diffusion_plan(
    cur: 'TrajectoryPoint',
    destination: 'TrajectoryPoint',
    obs):
    
    # DEBUG: bypass diffusion model, just return A* path
    from utils.map_process import _map_processor
    from utils.coord_utils import standard_to_carla
    
    if _map_processor._global_path is None:
        return []
    
    path = _map_processor._global_path  # (M, 2) standard frame
    
    trajectory = []
    for i in range(len(path)):
        sx, sy = path[i]
        carla_x, carla_y, _ = standard_to_carla(sx, sy, 0.0)
        
        if i == 0:
            speed = cur.speed
            angle = cur.angle
        else:
            prev_sx, prev_sy = path[i-1]
            dx = sx - prev_sx
            dy = sy - prev_sy
            import numpy as np
            angle_std = np.arctan2(dy, dx)
            _, _, carla_yaw_deg = standard_to_carla(0, 0, np.rad2deg(angle_std))
            angle = np.deg2rad(carla_yaw_deg)
            speed = max(MIN_SPEED, np.sqrt(dx**2 + dy**2) / 0.1)
        
        tp = TrajectoryPoint(Direction.FORWARD, carla_x, carla_y, speed, angle)
        trajectory.append(tp)
    
    print(f"[debug] Feeding A* path directly: {len(trajectory)} points")
    print(f"[debug] First wp: ({trajectory[0].x:.2f}, {trajectory[0].y:.2f})")
    print(f"[debug] Last wp: ({trajectory[-1].x:.2f}, {trajectory[-1].y:.2f})")
    print(f"[debug] Cur: ({cur.x:.2f}, {cur.y:.2f})")
    
    return trajectory

def diffusion_plan_real(
    cur: 'TrajectoryPoint',
    destination: 'TrajectoryPoint',
    obs):
    global _model

    ## failure handling

    if _model is None:
        print("[diffusion_adapter] WARNING: model not loaded, call load_model() first")
        return []


    try:
        ego_x, ego_y, ego_heading = carla_transform_to_standard(
            cur.x, cur.y, np.rad2deg(cur.angle)
        )
        anchor_ego_state = np.array([ego_x, ego_y, ego_heading], dtype=np.float64)

        # Set goal in ego-centric standard frame for guidance
        dest_x, dest_y, _ = carla_transform_to_standard(destination.x, destination.y, 0.0)
        anchor_matrix = _state_se2_array_to_transform_matrix(anchor_ego_state)
        anchor_inv = np.linalg.inv(anchor_matrix)
        dest_local = anchor_inv @ np.array([dest_x, dest_y, 1.0])
        _combined_guidance.set_goal(float(dest_local[0]), float(dest_local[1]))

        # Set A* path in ego-centric standard frame for path-following guidance
        from utils.map_process import _map_processor
        if _map_processor._global_path is not None:
            path_global = _map_processor._global_path  # (M, 2) standard frame
            # Transform to ego-centric: append 1s for homogeneous coords
            ones = np.ones((len(path_global), 1), dtype=np.float64)
            path_h = np.hstack([path_global, ones])  # (M, 3)
            path_ego = (anchor_inv @ path_h.T).T[:, :2]  # (M, 2)
            _combined_guidance.set_path(path_ego)

        inputs = _build_model_inputs(cur, obs)

        if _observation_normalizer is not None:
            inputs = _observation_normalizer(inputs)
        
        with torch.no_grad():
            _, outputs = _model(inputs)

        trajectory = _parse_model_output(outputs, anchor_ego_state, cur)

        return trajectory

    except Exception as e:
        print(f"[diffusion_adapter] Planning failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def init_scenario(cur: TrajectoryPoint, destination: TrajectoryPoint, obs):
    global _astar_path_set, _history_buffer
    _history_buffer = AgentHistoryBuffer()

    print("[diffusion_adapter] Running Hybrid A* for route prior...")
    astar_trajectory = plan_hybrid_a_star(cur, destination, obs)
 
    if astar_trajectory:
        set_astar_path(astar_trajectory)
        _astar_path_set = True
        print(f"[diffusion_adapter] Route prior set: {len(astar_trajectory)} waypoints")
        print(f"A* first point CARLA: ({astar_trajectory[0].x:.2f}, {astar_trajectory[0].y:.2f})")
        print(f"A* last point CARLA: ({astar_trajectory[-1].x:.2f}, {astar_trajectory[-1].y:.2f})")
        print(f"Cur CARLA: ({cur.x:.2f}, {cur.y:.2f})")
        print(f"Destination CARLA: ({destination.x:.2f}, {destination.y:.2f})")
        
    else:
        print("[diffusion_adapter] WARNING: Hybrid A* failed, route_lanes will be zeros")
        _astar_path_set = False






