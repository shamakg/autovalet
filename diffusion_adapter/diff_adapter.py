"""
diffusion_adapter.py  —  Clean rewrite with classifier guidance.

Coordinate conventions
----------------------
CARLA   : left-handed, Y points LEFT, yaw clockwise, angles in radians
Standard: right-handed, std_y = -CARLA_y, std_h = -CARLA_heading_rad

Conversion (coord_utils.py):
    std_x, std_y, std_h = carla_transform_to_standard(cx, cy, heading_deg)

Public API
----------
    load_model(ckpt_path, args_path)
    init_scenario(cur, destination, obs)   -- call once per scenario
    diffusion_plan(cur, destination, obs)  -- call every tick
"""

from __future__ import annotations

import os
import sys
import numpy as np
import torch
from typing import List, Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE          = os.path.dirname(os.path.abspath(__file__))
_AUTOVALET_ROOT = os.path.dirname(_HERE)
_NUPLAN_ROOT   = os.path.join(_HERE, "nuplan-devkit")
_DP_ROOT       = os.path.join(_HERE, "Diffusion-Planner")

for _p in [_HERE, _AUTOVALET_ROOT, _NUPLAN_ROOT, _DP_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Flush stale sub-modules so re-imports are clean
for _k in list(sys.modules):
    if _k.startswith("diffusion_planner.utils") or _k.startswith("diffusion_planner.model"):
        del sys.modules[_k]
# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from diffusion_planner.data_process.utils import (
    convert_to_model_inputs,
    vector_set_coordinates_to_local_frame,
    _state_se2_array_to_transform_matrix,
    _state_se2_array_to_transform_matrix_batch,
    _transform_matrix_to_state_se2_array_batch,
)

from utils.coord_utils import carla_transform_to_standard, standard_to_carla
from utils.map_process import LANE_NUM, LANE_LEN
from utils.agent_process import AgentHistoryBuffer, build_neighbor_agents_past
from v2 import TrajectoryPoint, Direction, MIN_SPEED, MAX_SPEED

# ---------------------------------------------------------------------------
# Guidance scales — CALIBRATED to match collision_guidance_fn magnitude
# ---------------------------------------------------------------------------
# The DPM update is:  Δx0_norm ≈ 0.5 × (σ²_t / α_t) × cond_grad  per element.
# At the first gated step (t≈0.49): σ≈0.956, α≈0.295 → factor ≈ 3.10.
# With state normaliser std=[20,20,1,1] the std cancels in our energy formulas:
#   pos_penalty gradient  = PATH_SCALE / T        per y-element  (T=80)
#   goal_energy gradient  = GOAL_SCALE            per element at final timestep
# Target: cond_grad ≈ 0.1–0.5 so step-1 correction ≈ 0.31–1.55 norm units (6–31 m).
# Too large (old 750/80=9.375) → 14.5 norm units → trajectory destroyed → no effect.
# Too small → not enough cumulative correction over 5 guided steps.
# Calibrated for ~8 m lateral correction over the 5 guided steps (t<0.5):
#   total_Δx0_norm = SCALE × 0.5 × Σ(σ²/α)_t  ≈  SCALE × 2.16
#   needed: 8 m / 20 m·norm⁻¹ = 0.4 norm  →  PATH_SCALE ≈ 0.4/2.16×80 ≈ 15
GOAL_SCALE  = 0.2    # pulls final waypoint toward destination (t < 0.5 gated)
PATH_SCALE  = 8.0    # penalises route/lateral deviation (t < 0.5 gated)
HALF_WIDTH  = 1.0    # route lane half-width in metres (nuPlan default is 1.75m)

# When the destination is mostly lateral (forward component < TURN_X_THRESHOLD metres),
# rotate the conditioning anchor to the parking-spot heading so the model sees the
# destination as straight ahead and generates an in-distribution forward trajectory.
TURN_X_THRESHOLD = 5.0   # metres — switch anchor when dest_ego_x < this.
# At 5 m before the junction, the ~5-6 m turning radius arc ends approximately
# at the junction row (y_end ≈ y_junction).  At 6 m the car arrives 1 m north of
# the junction, clipping the adjacent parked car on the approach.  At 3 m it
# overshoots the junction entirely.

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
_device: str = "cuda" if torch.cuda.is_available() else "cpu"
_model:  Optional[Diffusion_Planner] = None
_config: Optional[Config] = None
_observation_normalizer = None

_dest_std:         Optional[np.ndarray] = None   # (2,) destination in standard frame
_dest_angle_std:   Optional[float]      = None   # parking spot heading in std frame (radians)
_start_std:        Optional[np.ndarray] = None   # (2,) start in standard frame
_anchor_inv:       Optional[np.ndarray] = None   # (3,3) inverse SE2 of current ego pose
_route_pts_ego:    Optional[np.ndarray] = None   # (N,2) L-shaped route in ego-centric real metres
_suppress_guidance: bool                = False   # True in aisle: prevents premature East drift
_tick_log:          list               = []       # (x, y, heading_rad, dest_ego_x, suppress)


# ---------------------------------------------------------------------------
# Guidance function
# ---------------------------------------------------------------------------

def _guidance_fn(x, t, cond, **kwargs):
    """
    Classifier guidance energy for parking.

    Two phases:
      AISLE  (_suppress_guidance=True):  lateral position + velocity penalty → car goes straight.
      JUNCTION (_suppress_guidance=False): goal pull + route-proximity penalty → car turns into spot.

    Gradient calibration (state normaliser std=[20,20,1,1]):
      std cancels in pos/hdg formulas: cond_grad ≈ PATH_SCALE/T ≈ 0.1 per element.
      goal_energy: cond_grad ≈ GOAL_SCALE ≈ 0.2 at final timestep.
      Both give step-1 DPM correction ≈ 0.3–0.6 normalised units (6–12 m real) — stable.
    """
    zero = (x * 0.0).sum()

    if _dest_std is None or _anchor_inv is None:
        return zero

    t_val = float(t) if not isinstance(t, float) else t
    if not (0.005 < t_val < 0.5):
        return zero

    state_normalizer = kwargs.get("state_normalizer")
    model            = kwargs.get("model")
    model_condition  = kwargs.get("model_condition")
    if state_normalizer is None or model is None or model_condition is None:
        return zero

    B, P, D = x.shape

    # Tweedie correction — official GuidanceWrapper pattern (both sides detached).
    # x_fix is a constant; gradient flows only through x_in → corrected x.
    with torch.no_grad():
        x_pred = model(x, t, **model_condition)
    x_fix = x_pred.detach() - x.detach()
    x_fix = x_fix.reshape(B, P, -1, 4)
    x_fix[:, :, 0] = 0.0
    x = x + x_fix.reshape(B, P, -1)

    T_plus1 = D // 4
    mean_xy = state_normalizer.mean[0, 0, :2].to(x.device)   # [10, 0]
    std_xy  = state_normalizer.std [0, 0, :2].to(x.device)   # [20, 20]
    future_xy_norm = x[:, 0, :].reshape(B, T_plus1, 4)[:, 1:, :2]   # (B, T, 2)
    future_xy_real = future_xy_norm * std_xy[None, None, :] + mean_xy[None, None, :]

    if _suppress_guidance:
        # AISLE: penalise lateral displacement (y) and lateral velocity (dy).
        # Gradient per y-element: ±PATH_SCALE/T (std cancels: /std_x × std_y = 1).
        pos_penalty = -(PATH_SCALE / std_xy[0]) * torch.mean(
            torch.abs(future_xy_real[:, :, 1]))
        dy          = torch.diff(future_xy_real[:, :, 1], dim=1)
        hdg_penalty = -(PATH_SCALE / std_xy[0]) * torch.mean(torch.abs(dy))
        energy      = pos_penalty + hdg_penalty
        print(f"[guidance-aisle] t={t_val:.3f}  "
              f"pos={pos_penalty.item():.3f}  hdg={hdg_penalty.item():.3f}  "
              f"y_mean={future_xy_real[:,:,1].mean().item():.2f}m")
        return energy

    # JUNCTION: goal pull + path proximity.
    dest  = torch.tensor(_dest_std,   dtype=x.dtype, device=x.device)
    M_inv = torch.tensor(_anchor_inv, dtype=x.dtype, device=x.device)
    dest_h   = torch.stack([dest[0], dest[1],
                             torch.ones(1, device=x.device).squeeze(0)])
    dest_ego = (M_inv @ dest_h)[:2]

    final_real  = future_xy_real[:, -1, :]
    goal_dist   = torch.norm(final_real - dest_ego[None], dim=-1)
    goal_energy = -(GOAL_SCALE / std_xy[0]) * goal_dist.mean()

    if _route_pts_ego is not None and len(_route_pts_ego) >= 2:
        route = torch.tensor(_route_pts_ego, dtype=x.dtype, device=x.device)
        if len(route) > 200:
            idx   = torch.linspace(0, len(route) - 1, 200,
                                   dtype=torch.long, device=x.device)
            route = route[idx]
        diff_r        = future_xy_real[:, :, None, :] - route[None, None, :, :]
        dist_r        = torch.norm(diff_r, dim=-1)
        min_dist_r, _ = dist_r.min(dim=-1)
        path_energy   = -(PATH_SCALE / std_xy[0]) * min_dist_r.mean()
    else:
        path_energy = zero

    energy = goal_energy + path_energy
    print(f"[guidance-junc] t={t_val:.3f}  "
          f"goal={goal_energy.item():.3f}  path={path_energy.item():.3f}  "
          f"dest_ego=({dest_ego[0].item():.1f},{dest_ego[1].item():.1f})m")
    return energy


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, args_path: str) -> None:
    global _model, _config, _observation_normalizer

    if _model is not None:
        return

    print(f"[diffusion_adapter] Loading model from {ckpt_path}")
    _config = Config(args_path, guidance_fn=_guidance_fn)
    _observation_normalizer = _config.observation_normalizer

    _model = Diffusion_Planner(_config)
    raw = torch.load(ckpt_path, map_location=_device)
    sd  = raw.get("ema_state_dict") or raw.get("model") or raw
    sd  = {(k[len("module."):] if k.startswith("module.") else k): v
           for k, v in sd.items()}
    _model.load_state_dict(sd)
    _model.eval()
    _model = _model.to(_device)
    print(f"[diffusion_adapter] Model ready on {_device}  "
          f"(route_num={_config.route_num}, route_len={_config.route_len}, "
          f"predicted_neighbor_num={_config.predicted_neighbor_num})")


# ---------------------------------------------------------------------------
# Scenario init
# ---------------------------------------------------------------------------

def init_scenario(cur: TrajectoryPoint, destination: TrajectoryPoint, obs) -> None:
    global _dest_std, _dest_angle_std, _start_std, _suppress_guidance
    dest_x, dest_y, _ = carla_transform_to_standard(destination.x, destination.y, 0.0)
    _dest_std = np.array([dest_x, dest_y], dtype=np.float64)
    _dest_angle_std = float(destination.angle)   # radians; coord_utils is a passthrough
    start_x, start_y, _ = carla_transform_to_standard(cur.x, cur.y, 0.0)
    _start_std = np.array([start_x, start_y], dtype=np.float64)
    _suppress_guidance = True   # reset to aisle phase for each new scenario
    print(f"[diffusion_adapter] Route: "
          f"std({start_x:.1f},{start_y:.1f}) → std({dest_x:.1f},{dest_y:.1f}) "
          f"spot_angle={np.rad2deg(_dest_angle_std):.1f}°")


# ---------------------------------------------------------------------------
# Route lane builder — bent final-approach path matching VLA adapter style
# ---------------------------------------------------------------------------

def _sample_polyline(waypoints: list, n: int) -> np.ndarray:
    """Sample n evenly-spaced points along a piecewise-linear path."""
    pts   = np.array(waypoints, dtype=np.float64)
    dists = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    total = dists[-1]
    if total < 1e-6:
        return np.tile(pts[0], (n, 1))
    ts = np.linspace(0.0, total, n)
    return np.column_stack([np.interp(ts, dists, pts[:, 0]),
                            np.interp(ts, dists, pts[:, 1])])


def _sample_polyline_dense(waypoints: list, spacing: float = 0.5) -> np.ndarray:
    """Sample a piecewise-linear path at a fixed arc-length spacing."""
    pts   = np.array(waypoints, dtype=np.float64)
    dists = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    total = dists[-1]
    if total < 1e-6:
        return pts[:1]
    n  = max(2, int(total / spacing) + 1)
    ts = np.linspace(0.0, total, n)
    return np.column_stack([np.interp(ts, dists, pts[:, 0]),
                            np.interp(ts, dists, pts[:, 1])])


def _make_straight_route_lanes(anchor: np.ndarray) -> np.ndarray:
    """
    Build (route_num, route_len, 12) route lane tensor matching the nuPlan
    training data structure.

    With the bearing anchor always active, the route is always a straight line
    from ego toward the destination (plus overshoot).  The bearing heading already
    tells the model "destination = straight ahead", so an L-shaped route is not
    needed and can confuse the model into turning prematurely.
    """
    route_num = _config.route_num
    route_len = _config.route_len

    ego_xy = anchor[:2]
    if _dest_std is None:
        return np.zeros((route_num, route_len, 12), dtype=np.float32)

    # Always: straight line from ego to destination, then overshoot
    diff = _dest_std - ego_xy
    dist = np.linalg.norm(diff)
    if dist < 1e-3:
        return np.zeros((route_num, route_len, 12), dtype=np.float32)
    unit      = diff / dist
    overshoot = _dest_std + 45.0 * unit
    waypoints = [ego_xy, _dest_std, overshoot]

    # Dense global points along the route
    pts_global = _sample_polyline_dense(waypoints, spacing=0.5)   # (N, 2) global

    # Convert all to ego-centric in one shot
    N      = len(pts_global)
    avails = np.ones((1, N), dtype=np.bool_)
    pts_ego = vector_set_coordinates_to_local_frame(
        pts_global[np.newaxis], avails, anchor
    )[0]   # (N, 2)

    global _route_pts_ego
    _route_pts_ego = pts_ego.copy()

    # Use overlapping sliding windows so all route_num segments are filled
    # from a short (~50m) in-distribution path.
    # stride = max(1, (N - route_len) / (route_num - 1))
    route_lanes = np.zeros((route_num, route_len, 12), dtype=np.float32)
    tl = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (route_len, 1))

    if N <= route_len:
        n_windows = 1
        stride    = 0
    else:
        n_windows = min(route_num, N - route_len + 1)
        stride    = max(1, (N - route_len) // max(1, n_windows - 1))

    for i in range(n_windows):
        start = i * stride
        seg   = pts_ego[start : start + route_len]
        n     = len(seg)
        if n < route_len:
            seg = np.vstack([seg, np.tile(seg[-1], (route_len - n, 1))])

        # Raw step diffs (~0.5m magnitude) — NOT unit vectors.
        # Training _lane_polyline_process uses polyline[i+1]-polyline[i] directly.
        fwd_raw   = np.diff(seg, axis=0)
        fwd_raw   = np.vstack([fwd_raw, fwd_raw[-1:]])

        # Unit perpendiculars for boundary offsets (features 4-7)
        norms      = np.linalg.norm(fwd_raw, axis=1, keepdims=True)
        safe_norms = np.where(norms < 1e-6, 1.0, norms)
        left_off   = HALF_WIDTH * np.column_stack(
            [-fwd_raw[:, 1],  fwd_raw[:, 0]]) / safe_norms   # (N,2) CCW = left
        right_off  = HALF_WIDTH * np.column_stack(
            [ fwd_raw[:, 1], -fwd_raw[:, 0]]) / safe_norms   # (N,2) CW  = right

        route_lanes[i] = np.concatenate([
            seg.astype(np.float32),
            fwd_raw.astype(np.float32),      # [2:4] raw step diffs, ~0.5m
            left_off.astype(np.float32),     # [4:6] left boundary offset from centre
            right_off.astype(np.float32),    # [6:8] right boundary offset from centre
            tl,                              # [8:12] TL one-hot [unknown]
        ], axis=1)

    return route_lanes


# ---------------------------------------------------------------------------
# Build model inputs
# ---------------------------------------------------------------------------

def _build_model_inputs(
    cur: TrajectoryPoint,
    anchor: np.ndarray,
    history_buffer: Optional[AgentHistoryBuffer] = None,
) -> dict:
    route_lanes = _make_straight_route_lanes(anchor)
    lanes = np.zeros((LANE_NUM, LANE_LEN, 12), dtype=np.float32)

    if history_buffer is not None:
        neighbor_agents = build_neighbor_agents_past(history_buffer, anchor).astype(np.float32)
    else:
        neighbor_agents = np.zeros((32, 21, 11), dtype=np.float32)

    data = {
        "neighbor_agents_past":         neighbor_agents,
        "ego_current_state":            np.array(
            [0., 0., 1., 0., float(cur.speed), 0., 0., 0., 0., 0.],
            dtype=np.float32,
        ),
        "static_objects":               np.zeros((5, 10),  dtype=np.float32),
        "lanes":                        lanes,
        "lanes_speed_limit":            np.full((LANE_NUM, 1),            MAX_SPEED, dtype=np.float32),
        "lanes_has_speed_limit":        np.ones ((LANE_NUM, 1),            dtype=np.bool_),
        "route_lanes":                  route_lanes,
        "route_lanes_speed_limit":      np.full((_config.route_num, 1),   MAX_SPEED, dtype=np.float32),
        "route_lanes_has_speed_limit":  np.ones ((_config.route_num, 1),  dtype=np.bool_),
    }
    return convert_to_model_inputs(data, _device)


# ---------------------------------------------------------------------------
# Parse model output → CARLA trajectory
# ---------------------------------------------------------------------------

def _parse_model_output(
    outputs: dict,
    anchor:  np.ndarray,
    cur:     TrajectoryPoint,
) -> List[TrajectoryPoint]:
    """
    outputs['prediction']: (B, P, T, 4) — ego is [:, 0, :, :], already denormalized
    columns: [x, y, cos_h, sin_h] in ego-centric standard frame
    """
    preds = outputs["prediction"][0, 0].detach().cpu().numpy().astype(np.float64)
    T     = preds.shape[0]

    # Normalize heading vector before arctan2 (linear denorm doesn't enforce unit circle)
    cos_h = preds[:, 2];  sin_h = preds[:, 3]
    norm  = np.sqrt(cos_h**2 + sin_h**2)
    norm  = np.where(norm < 1e-6, 1.0, norm)
    headings    = np.arctan2(sin_h / norm, cos_h / norm)
    local_poses = np.stack([preds[:, 0], preds[:, 1], headings], axis=1)

    # Ego-centric standard → global standard → CARLA
    anchor_mat   = _state_se2_array_to_transform_matrix(anchor)
    pose_mats    = _state_se2_array_to_transform_matrix_batch(local_poses)
    global_poses = _transform_matrix_to_state_se2_array_batch(anchor_mat @ pose_mats)

    trajectory: List[TrajectoryPoint] = []
    DT = 0.1

    for i in range(T):
        gx, gy, gh   = global_poses[i]
        cx, cy, cdeg = standard_to_carla(gx, gy, gh)

        if i == 0:
            speed = max(MIN_SPEED, float(cur.speed))
        else:
            prev  = global_poses[i - 1, :2]
            speed = max(MIN_SPEED, np.linalg.norm([gx - prev[0], gy - prev[1]]) / DT)

        trajectory.append(
            TrajectoryPoint(Direction.FORWARD, cx, cy, speed, np.deg2rad(cdeg))
        )

    _log(anchor, preds, global_poses, trajectory, cur)
    return trajectory


def _log(anchor, preds, global_poses, traj, cur):
    if _dest_std is not None:
        dest_cx, dest_cy, _ = standard_to_carla(_dest_std[0], _dest_std[1], 0.0)
    else:
        dest_cx = dest_cy = float("nan")
    print(f"[adapter] anchor std: ({anchor[0]:.1f},{anchor[1]:.1f},{np.rad2deg(anchor[2]):.1f}°)")
    print(f"[adapter] pred[0]  ego=({preds[0,0]:.2f},{preds[0,1]:.2f})  "
          f"CARLA=({traj[0].x:.2f},{traj[0].y:.2f})")
    print(f"[adapter] pred[-1] ego=({preds[-1,0]:.2f},{preds[-1,1]:.2f})  "
          f"CARLA=({traj[-1].x:.2f},{traj[-1].y:.2f})")
    print(f"[adapter] dest CARLA=({dest_cx:.1f},{dest_cy:.1f})  "
          f"cur CARLA=({cur.x:.2f},{cur.y:.2f})")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def diffusion_plan(
    cur:            TrajectoryPoint,
    destination:    TrajectoryPoint,
    obs,
    history_buffer: Optional[AgentHistoryBuffer] = None,
) -> List[TrajectoryPoint]:
    global _anchor_inv, _suppress_guidance, _tick_log

    if _model is None:
        print("[diffusion_adapter] ERROR: call load_model() first")
        return []
    if _dest_std is None:
        print("[diffusion_adapter] ERROR: call init_scenario() first")
        return []

    try:
        std_x, std_y, std_h = carla_transform_to_standard(
            cur.x, cur.y, np.rad2deg(cur.angle)
        )
        anchor = np.array([std_x, std_y, std_h], dtype=np.float64)

        # Always use a bearing anchor: rotate the model's ego frame so that
        # "straight ahead" (East in nuPlan) points directly at the destination.
        # This makes the model's strong East prior work FOR us — instead of
        # fighting it with guidance that can't overcome the OOD Jacobian ≈ 0.
        dx_to_dest   = _dest_std[0] - std_x
        dy_to_dest   = _dest_std[1] - std_y
        dist_to_dest = np.hypot(dx_to_dest, dy_to_dest)
        if dist_to_dest > 0.5:
            effective_h = float(np.arctan2(dy_to_dest, dx_to_dest))
        else:
            effective_h = _dest_angle_std if _dest_angle_std is not None else std_h

        # Phase switch: close to destination → goal-pull guidance (junction).
        # Far → lateral-centering guidance (aisle) to keep car on bearing line.
        # One-way latch: never revert from junction back to aisle.
        if _suppress_guidance:
            _suppress_guidance = (dist_to_dest >= TURN_X_THRESHOLD)
        _tick_log.append((std_x, std_y, std_h, dist_to_dest, _suppress_guidance))
        print(f"[adapter] dist_to_dest={dist_to_dest:.1f}m  bearing={np.rad2deg(effective_h):.1f}°  "
              f"suppress_guidance={_suppress_guidance}")

        effective_anchor = np.array([std_x, std_y, effective_h], dtype=np.float64)
        M                = _state_se2_array_to_transform_matrix(effective_anchor)
        _anchor_inv      = np.linalg.inv(M)

        inputs = _build_model_inputs(cur, effective_anchor, history_buffer)
        if _observation_normalizer is not None:
            inputs = _observation_normalizer(inputs)

        with torch.no_grad():
            _, outputs = _model(inputs)

        return _parse_model_output(outputs, effective_anchor, cur)

    except Exception as exc:
        import traceback
        print(f"[diffusion_adapter] Planning failed: {exc}")
        traceback.print_exc()
        return []