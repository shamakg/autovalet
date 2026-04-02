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
import carla

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
    _global_state_se2_array_to_local,
    _global_velocity_to_local,
)

from utils.coord_utils import (
    carla_transform_to_standard,
    carla_velocity_to_standard,
    standard_to_carla,
)
from utils.agent_process import (
    AgentHistoryBuffer,
    AgentState,
    NUM_AGENTS,
    NUM_PAST_STEPS,
    build_neighbor_agents_past,
)
from utils.guidance import _CombinedGuidance, _CombinedGuidance_v4, _CombinedGuidance_v3
from v2 import TrajectoryPoint, Direction, MIN_SPEED, MAX_SPEED, plan_hybrid_a_star
from utils.map_process import build_map_features, set_astar_path, LANE_NUM, ROUTE_LANE_NUM, get_dist
from utils.debug import (
    _draw_ego_frame_debug,
    _draw_route_lanes_debug,
    debug_agents,
    debug_ego_axes,
)


_combined_guidance = _CombinedGuidance_v4()
debug = False


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


    _config = Config(args_path, guidance_fn=_combined_guidance)
    _observation_normalizer = _config.observation_normalizer

    _model = Diffusion_Planner(_config)
    state_dict = torch.load(ckpt_path, map_location=_device)

    ### handle different modelf ormats
    if 'ema_state_dict' in state_dict:
        state_dict = state_dict['ema_state_dict']
    elif 'model' in state_dict:
        state_dict = state_dict['model']

    # remove module prefix (some bookkeeping)
    state_dict = {
        (k[len('module.'):] if k.startswith('module.') else k): v
        for k, v in state_dict.items()
    }

    # load weights
    _model.load_state_dict(state_dict)
    _model.eval()
    _model = _model.to(_device)

    print(f"[diffusion_adapter] Model loaded on {_device}")


def _build_agent_feature(state: AgentState, anchor_ego_state: np.ndarray, x0, y0, h) -> np.ndarray:
    # Position in ego frame using world_to_ego
    ex, ey = world_to_ego(state.x, state.y, x0, y0, h)
    
    # Heading in ego frame
    local_heading = state.heading - h
    
    # Velocity in ego frame
    vx_ego =  np.cos(h) * state.vx + np.sin(h) * state.vy
    vy_ego = -np.sin(h) * state.vx + np.cos(h) * state.vy

    type_vec = [1.0, 0.0, 0.0]  # vehicle by default
    if state.agent_type == 'pedestrian':
        type_vec = [0.0, 1.0, 0.0]

    return np.array([
        ex, ey,
        np.cos(local_heading), np.sin(local_heading),
        vx_ego, vy_ego,
        ### slightly overestimate vehicle dimensions
        state.width, state.length,
        *type_vec
    ], dtype=np.float32)


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
    h = anchor_ego_state[2]
    x0, y0 = anchor_ego_state[0], anchor_ego_state[1]
    agent_states = []

    # Collect parked car IDs so we don't double-count them (they'll be passed separately)
    parked_car_ids = {v.id for v in obs.parked_cars} if hasattr(obs, 'parked_cars') else set()

    if hasattr(obs, 'dyn_obs_clusters'):
        for actor_id, cluster in obs.dyn_obs_clusters.items():
            if actor_id in parked_car_ids:
                continue
            mean = cluster['state_mean']  # [x, y, vx, vy] in CARLA global frame
            bb   = cluster['bb']
            # Derive heading from velocity; fall back to actual CARLA yaw if stationary
            vx_carla, vy_carla = mean[2], mean[3]
            if abs(vx_carla) > 0.01 or abs(vy_carla) > 0.01:
                yaw = np.arctan2(vy_carla, vx_carla)
            else:
                yaw = cluster.get('yaw', 0.0)  # use true orientation for parked cars

            # Account for bounding box center offset
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)
            bb_offset_x = bb.location.x * cos_y - bb.location.y * sin_y
            bb_offset_y = bb.location.x * sin_y + bb.location.y * cos_y
            actual_x = mean[0] + bb_offset_x
            actual_y = mean[1] + bb_offset_y

            std_x, std_y, std_heading = carla_transform_to_standard(actual_x, actual_y, np.rad2deg(yaw))
            std_vx, std_vy = carla_velocity_to_standard(vx_carla, vy_carla)

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


    agent_states.append(AgentState(
        actor_id = -1,
        x = ego_x,
        y = ego_y,
        heading = ego_heading,
        vx = std_vx,
        vy = std_vy,
        width = 2.0, # Standard car width
        length = 4.5, # Standard car length
        agent_type = 'vehicle'
    ))

    # Update history buffer with dynamic agents
    _history_buffer.update(agent_states)

    # Build neighbor_agents_past via training-pipeline function (distance sort, ped cap, SE(2))
    parked_cars = getattr(obs, 'parked_cars', None)
    neighbor_agents_past = build_neighbor_agents_past(
        _history_buffer, anchor_ego_state, parked_cars=parked_cars
    ).astype(np.float32)

    # print(f"[agents] Non-zero slots: {(np.abs(neighbor_agents_past).sum(axis=(1,2)) > 0).sum()} / {NUM_AGENTS}")
    
    for slot_idx in range(NUM_AGENTS):
        last_step = neighbor_agents_past[slot_idx, -1]
        if np.abs(last_step).sum() < 1e-6:
            continue
        ex, ey = last_step[0], last_step[1]
        vx, vy = last_step[4], last_step[5]
        w, l = last_step[6], last_step[7]
        speed = np.sqrt(vx**2 + vy**2)
        # Transform back to world for visualization
        wx, wy = ego_to_world(ex, ey, x0, y0, h)
        cx, cy, _ = standard_to_carla(wx, wy, 0.0)
        # print(f"  agent[{slot_idx}]: ego=({ex:.1f},{ey:.1f}) world=({cx:.1f},{cy:.1f}) speed={speed:.2f} size=({w:.1f}x{l:.1f})")

    # --- Static objects (none from ObstacleMap, leave zeros) ---
    # static_objects = build_static_objects()
    static_objects = np.zeros((5, 10), dtype=np.float32)

    lanes, route_lanes = build_map_features(anchor_ego_state)

    # print("Route lane 0 points (ego frame x,y):", route_lanes[0, :5, :2])

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
        # 'neighbor_agents_past': neighbor_agents_past[:, -21:],  # last 21 steps
        'neighbor_agents_past': neighbor_agents_past[:, -21:],
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

    # print(f"Raw predictions x: min={predictions[:, 0].min():.2f}, max={predictions[:, 0].max():.2f}")
    # print(f"Raw predictions y: min={predictions[:, 1].min():.2f}, max={predictions[:, 1].max():.2f}")

    T = predictions.shape[0]

    headings = np.arctan2(predictions[:, 3], predictions[:, 2])  # (T,)
    local_poses = np.stack([predictions[:, 0], predictions[:, 1], headings], axis=1)  # (T, 3)

    # print(f"local_poses[0]: x={local_poses[0,0]:.2f}, y={local_poses[0,1]:.2f}")
    # print(f"local_poses[-1]: x={local_poses[-1,0]:.2f}, y={local_poses[-1,1]:.2f}")

    # SE(2) composition: anchor @ local → global
    
    h0 = anchor_ego_state[2]
    x0, y0 = anchor_ego_state[0], anchor_ego_state[1]
    global_x = x0 + np.cos(h0) * local_poses[:, 0] - np.sin(h0) * local_poses[:, 1]
    global_y = y0 + np.sin(h0) * local_poses[:, 0] + np.cos(h0) * local_poses[:, 1]
    global_heading = h0 + local_poses[:, 2]
    global_poses = np.stack([global_x, global_y, global_heading], axis=1)

    # print(f"anchor: ({anchor_ego_state[0]:.2f}, {anchor_ego_state[1]:.2f}, h={np.rad2deg(anchor_ego_state[2]):.1f}°)")
    # print(f"global[0]: ({global_poses[0,0]:.2f}, {global_poses[0,1]:.2f})")
    # print(f"global[-1]: ({global_poses[-1,0]:.2f}, {global_poses[-1,1]:.2f})")

    trajectory = []
    for i in range(T):
        gx, gy, gh = global_poses[i]
        carla_x, carla_y, carla_yaw_deg = standard_to_carla(gx, gy, gh)
        carla_heading_rad = np.deg2rad(carla_yaw_deg)

        if i == 0:
            speed = MIN_SPEED
        else:
            prev_gx, prev_gy = global_poses[i-1, :2]
            dt = 0.1
            speed = max(MIN_SPEED, np.sqrt((gx-prev_gx)**2 + (gy-prev_gy)**2) / dt)

        tp = TrajectoryPoint(Direction.FORWARD, carla_x, carla_y, speed, carla_heading_rad)
        trajectory.append(tp)

    # print(f"First CARLA point: ({trajectory[0].x:.2f}, {trajectory[0].y:.2f})")
    # print(f"Last CARLA point: ({trajectory[-1].x:.2f}, {trajectory[-1].y:.2f})")
    # print(f"Cur: ({cur.x:.2f}, {cur.y:.2f})")

    return trajectory
    


def diffusion_plan( cur, destination, obs, world):
    global _model

    ## failure handling

    if _model is None:
        print("[diffusion_adapter] WARNING: model not loaded, call load_model() first")
        return []


    try:
        ego_x, ego_y, ego_heading = carla_transform_to_standard(cur.x, cur.y, np.rad2deg(cur.angle))
        anchor_ego_state = np.array([ego_x, ego_y, ego_heading], dtype=np.float64)

        dx_world = destination.x - cur.x  # positive = east
        dy_world = destination.y - cur.y  # positive = north
        # print(f"dest relative to car: dx={dx_world:.2f} (east+), dy={dy_world:.2f} (north+)")
        # print(f"car heading: {np.rad2deg(cur.angle):.1f} degrees")

        dest_x, dest_y, _ = carla_transform_to_standard(destination.x, destination.y, 0.0)
        h = anchor_ego_state[2]
        x0, y0 = anchor_ego_state[0], anchor_ego_state[1]

        dest_ego_x, dest_ego_y = world_to_ego(dest_x, dest_y, x0, y0, h)
        dist_to_goal = np.sqrt(dest_ego_x**2 + dest_ego_y**2)
        if dist_to_goal > 2.5:
            _combined_guidance.set_goal(float(dest_ego_x), float(dest_ego_y))
        else:
            _combined_guidance._goal = None  # disable goal guidance at close range to avoid arc artifacts

        from utils.map_process import _map_processor
        path_ego = None
        if _map_processor._global_path is not None:
            path_global = _map_processor._global_path
            dists = np.linalg.norm(path_global - np.array([x0, y0]), axis=1)
            closest = int(np.argmin(dists))
            path_remaining = path_global[closest:]
            path_ego = np.array([world_to_ego(p[0], p[1], x0, y0, h) for p in path_remaining])
            _combined_guidance.set_path(path_ego)

        inputs = _build_model_inputs(cur, obs)

        neighbor_np = inputs['neighbor_agents_past'][0].cpu().numpy()  # (32, 21, 11)

        
        
        route_lanes_np = inputs['route_lanes'][0].cpu().numpy()
        for lane_idx in range(route_lanes_np.shape[0]):
            lane = route_lanes_np[lane_idx]
            if np.abs(lane[:, :2]).sum() < 1e-6:
                continue

            ### Draw Hybrid A* Path Prior for debug
            if debug:
                for pt_idx in range(0, lane.shape[0], 2):
                    ego_x, ego_y = float(lane[pt_idx, 0]), float(lane[pt_idx, 1])
                    wx, wy = ego_to_world(ego_x, ego_y, x0, y0, h)
                    cx, cy, _ = standard_to_carla(wx, wy, 0.0)
                    world.debug.draw_point(
                        carla.Location(x=cx, y=cy, z=0.5),
                        size=0.1,
                        color=carla.Color(r=255, g=0, b=0),
                        life_time=1.0
                    )
       

        if _observation_normalizer is not None:
            inputs = _observation_normalizer(inputs)

        with torch.no_grad():
            _, outputs = _model(inputs)

        trajectory = _parse_model_output(outputs, anchor_ego_state, cur)

        dest_ego = np.array([dest_ego_x, dest_ego_y])
    

        if path_ego is not None and debug:
            debug_agents(world, neighbor_np, x0, y0, h)
            debug_ego_axes(world, anchor_ego_state)
            
            # _draw_route_lanes_debug(world, anchor_ego_state, inputs['route_lanes'][0])

        for tp in trajectory:
            tp.speed = np.clip(tp.speed, MIN_SPEED, MAX_SPEED)

        from diffusion_planner.data_process.utils import vector_set_coordinates_to_local_frame
        test_pt = np.array([[[dest_x, dest_y]]], dtype=np.float64)
        test_avail = np.ones((1, 1), dtype=np.bool_)
        test_ego = vector_set_coordinates_to_local_frame(test_pt, test_avail, anchor_ego_state)[0][0]

        return trajectory

    except Exception as e:
        print(f"[diffusion_adapter] Planning failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def init_scenario(cur: TrajectoryPoint, destination: TrajectoryPoint, obs):
    global _astar_path_set, _history_buffer
    _history_buffer = AgentHistoryBuffer()

    from utils.map_process import set_destination
    set_destination(destination.x, destination.y)

    print("[diffusion_adapter] Running Hybrid A* for route prior...")
    astar_trajectory = plan_hybrid_a_star(cur, destination, obs)
 
    if astar_trajectory:
        set_astar_path(astar_trajectory)
        _astar_path_set = True
        
    else:
        print("[diffusion_adapter] WARNING: Hybrid A* failed, route_lanes will be zeros")
        _astar_path_set = False

def world_to_ego(wx, wy, x0, y0, h):
    dx, dy = wx - x0, wy - y0
    # Standard Right-Handed Rotation Matrix
    ex =  np.cos(h) * dx + np.sin(h) * dy
    ey = -np.sin(h) * dx + np.cos(h) * dy
    # If your 'h' comes from carla_transform_to_standard, 
    # it is already CCW radians. To make Left = +Y:
    return ex, ey # Ensure your math here results in Left being positive

def ego_to_world(ex, ey, x0, y0, h):
    # Inverse: Rotate by +h then translate
    wx = x0 + np.cos(h) * ex - np.sin(h) * ey
    wy = y0 + np.sin(h) * ex + np.cos(h) * ey
    return wx, wy
