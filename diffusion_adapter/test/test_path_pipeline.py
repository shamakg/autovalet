"""
test_path_pipeline.py — End-to-end test of the coordinate pipeline from model
output through to pixel coordinates on the top-down camera.

Tests each stage independently so we can pinpoint exactly where the path collapses:
  Stage 1: Model output is spread (with guidance + route lanes, matching live run)
  Stage 2: ego→world transform preserves spread
  Stage 3: latest_pred_route stored correctly (world coords, not re-ego'd)
  Stage 4: project_world_to_topdown maps to distinct pixels

Run with:
    /home/sumesh/envs/simlingo/bin/python test/test_path_pipeline.py
"""

import sys, os
import numpy as np
import torch

_HERE         = os.path.dirname(os.path.abspath(__file__))
_ADAPTER_ROOT = os.path.dirname(_HERE)
_AUTOVALET    = os.path.dirname(_ADAPTER_ROOT)
_NUPLAN_ROOT  = os.path.join(_ADAPTER_ROOT, "nuplan-devkit")
_DP_ROOT      = os.path.join(_ADAPTER_ROOT, "Diffusion-Planner")
_CARLA_ROOT   = "/home/sumesh/opt/carla/PythonAPI/carla"
_SCENARIO_ROOT= "/home/sumesh/carla_garage/scenario_runner"
_LB_ROOT      = "/home/sumesh/carla_garage/leaderboard"

for _p in [_CARLA_ROOT, _SCENARIO_ROOT, _LB_ROOT,
           _HERE, _AUTOVALET, _ADAPTER_ROOT, _NUPLAN_ROOT, _DP_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from diffusion_planner.data_process.utils import (
    convert_to_model_inputs,
    _state_se2_array_to_transform_matrix,
    _state_se2_array_to_transform_matrix_batch,
    _transform_matrix_to_state_se2_array_batch,
)
from utils.coord_utils import carla_transform_to_standard, standard_to_carla
from utils.map_process import LANE_NUM
from utils.agent_process import AgentHistoryBuffer, AgentState, build_neighbor_agents_past

CKPT = os.path.join(_ADAPTER_ROOT, "checkpoints/model.pth")
ARGS = os.path.join(_ADAPTER_ROOT, "checkpoints/args.json")

# Exact values from the live run logs
EGO_CX,  EGO_CY,  EGO_YAW_DEG = 285.60, -243.73, 90.0
DEST_CX, DEST_CY               = 293.3,  -223.4
CAM_WIDTH, CAM_HEIGHT, CAM_Z   = 480, 320, 30

GOAL_SCALE = 5000.0
PATH_SCALE = 2500.0


def ok(tag, d=""):   print(f"  [PASS] {tag}")
def fail(tag, d=""): print(f"  [FAIL] {tag}  {d}")
def check(cond, tag, detail=""):
    (ok if cond else fail)(tag, detail)
    return cond


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_model(gfn, device):
    config = Config(ARGS, guidance_fn=gfn)
    model  = Diffusion_Planner(config)
    raw = torch.load(CKPT, map_location=device)
    sd  = raw.get("ema_state_dict") or raw.get("model") or raw
    sd  = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model.to(device), config


def _make_guidance_fn(dest_ego_np, goal_scale=GOAL_SCALE, path_scale=PATH_SCALE):
    dest_t = torch.tensor(dest_ego_np, dtype=torch.float32)

    def guidance_fn(x, t, cond, **kwargs):
        zero         = (x * 0.0).sum()
        model        = kwargs.get("model")
        mc           = kwargs.get("model_condition")
        state_norm   = kwargs.get("state_normalizer")   # set by DPM solver, has .mean/.std
        if model is None or mc is None or state_norm is None:
            print(f"  [guidance] missing kwargs: model={model is not None} "
                  f"mc={mc is not None} state_norm={state_norm is not None}")
            return zero

        B, P, D = x.shape
        with torch.no_grad():
            x_fix = model(x, t, **mc) - x.detach()
        x_fix = x_fix.reshape(B, P, -1, 4)
        x_fix[:, :, 0] = 0.0
        x = x + x_fix.reshape(B, P, -1)

        T1          = D // 4
        future_norm = x[:, 0, :].reshape(B, T1, 4)[:, 1:, :2]
        dest        = dest_t.to(x.device)
        mean_       = state_norm.mean[0, 0, :2].to(x.device)
        std_        = state_norm.std [0, 0, :2].to(x.device)
        future      = future_norm * std_[None, None, :] + mean_[None, None, :]

        goal_energy = -(goal_scale / std_[0]) * torch.norm(future[:, -1, :] - dest[None], dim=-1).mean()

        d = torch.norm(dest)
        if d > 1e-3:
            unit        = dest / d.detach()
            path_energy = (path_scale / std_[0]) * (future * unit[None, None, :]).sum(-1).mean()
        else:
            path_energy = zero

        return goal_energy + path_energy

    return guidance_fn


def _build_route_lanes(config, dest_ego_np, spacing=0.5, half_width=1.75):
    route_num, route_len = config.route_num, config.route_len
    dest  = np.array(dest_ego_np, dtype=np.float64)
    dist  = np.linalg.norm(dest)
    if dist < 1e-3:
        return np.zeros((route_num, route_len, 12), dtype=np.float32)

    unit      = dest / dist
    overshoot = dest + 45.0 * unit
    pts       = np.array([np.zeros(2), dest, overshoot])
    cdists    = np.concatenate([[0.], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    total     = cdists[-1]
    n         = max(2, int(total / spacing) + 1)
    ts        = np.linspace(0., total, n)
    pts_ego   = np.column_stack([np.interp(ts, cdists, pts[:, 0]),
                                  np.interp(ts, cdists, pts[:, 1])])

    tl          = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (route_len, 1))
    route_lanes = np.zeros((route_num, route_len, 12), dtype=np.float32)
    N  = len(pts_ego)
    nw = min(route_num, N - route_len + 1) if N > route_len else 1
    st = max(1, (N - route_len) // max(1, nw - 1)) if N > route_len else 0

    for i in range(nw):
        seg = pts_ego[i*st : i*st + route_len]
        if len(seg) < route_len:
            seg = np.vstack([seg, np.tile(seg[-1], (route_len - len(seg), 1))])
        fwd = np.diff(seg, axis=0); fwd = np.vstack([fwd, fwd[-1:]])
        nm  = np.linalg.norm(fwd, axis=1, keepdims=True)
        nm  = np.where(nm < 1e-6, 1., nm)
        lo  = half_width * np.column_stack([-fwd[:, 1],  fwd[:, 0]]) / nm
        ro  = half_width * np.column_stack([ fwd[:, 1], -fwd[:, 0]]) / nm
        route_lanes[i] = np.concatenate([
            seg.astype(np.float32), fwd.astype(np.float32),
            lo.astype(np.float32),  ro.astype(np.float32), tl,
        ], axis=1)
    return route_lanes


def _make_inputs(device, config, route_lanes):
    MAX_SPEED = 14.0
    data = {
        "neighbor_agents_past":        np.zeros((32, 21, 11), dtype=np.float32),
        "ego_current_state":           np.array([0.,0.,1.,0.,1.5,0.,0.,0.,0.,0.], dtype=np.float32),
        "static_objects":              np.zeros((5, 10), dtype=np.float32),
        "lanes":                       np.zeros((LANE_NUM, 20, 12), dtype=np.float32),
        "lanes_speed_limit":           np.full((LANE_NUM, 1), MAX_SPEED, dtype=np.float32),
        "lanes_has_speed_limit":       np.ones((LANE_NUM, 1), dtype=np.bool_),
        "route_lanes":                 route_lanes,
        "route_lanes_speed_limit":     np.full((config.route_num, 1), MAX_SPEED, dtype=np.float32),
        "route_lanes_has_speed_limit": np.ones((config.route_num, 1), dtype=np.bool_),
    }
    return convert_to_model_inputs(data, device)


def _ego_to_world(pred_xy, ego_cx, ego_cy, carla_yaw_rad):
    """Mirror of recording_utils.ego_to_world."""
    c, s = np.cos(carla_yaw_rad), np.sin(carla_yaw_rad)
    world = np.empty_like(pred_xy)
    world[:, 0] = ego_cx + pred_xy[:, 0] * c - pred_xy[:, 1] * s
    world[:, 1] = ego_cy + pred_xy[:, 0] * s + pred_xy[:, 1] * c
    return world


def _project(world_x, world_y, actor_cx, actor_cy):
    """Mirror of recording_utils.project_world_to_topdown."""
    visible = 2 * CAM_Z
    px = int(CAM_WIDTH  / 2 - (world_x - actor_cx) * (CAM_WIDTH  / visible))
    py = int(CAM_HEIGHT / 2 - (world_y - actor_cy) * (CAM_HEIGHT / visible))
    return px, py


def compute_dest_ego():
    """Compute destination in ego-centric standard frame."""
    std_x, std_y, std_h = carla_transform_to_standard(EGO_CX, EGO_CY, EGO_YAW_DEG)
    dest_sx, dest_sy, _ = carla_transform_to_standard(DEST_CX, DEST_CY, 0.0)
    anchor  = np.array([std_x, std_y, std_h])
    M_inv   = np.linalg.inv(_state_se2_array_to_transform_matrix(anchor))
    dest_ego = (M_inv @ np.array([dest_sx, dest_sy, 1.0]))[:2]
    return anchor, dest_ego


# ---------------------------------------------------------------------------
# Stage 1 — Model output spread (guidance + route lanes, matching live run)
# ---------------------------------------------------------------------------

def test_stage1_model_output_spread(device):
    print("\n=== Stage 1: Model output spread (guidance + route lanes) ===")

    anchor, dest_ego = compute_dest_ego()
    print(f"  dest in ego frame: ({dest_ego[0]:.2f}, {dest_ego[1]:.2f})  "
          f"dist={np.linalg.norm(dest_ego):.2f}m")

    tmp_config = Config(ARGS, guidance_fn=None)
    gfn        = _make_guidance_fn(dest_ego)
    model, config = _load_model(gfn, device)

    route_lanes = _build_route_lanes(config, dest_ego)
    inputs      = _make_inputs(device, config, route_lanes)
    if config.observation_normalizer:
        inputs = config.observation_normalizer(inputs)

    with torch.no_grad():
        _, outputs = model(inputs)

    preds    = outputs["prediction"][0, 0].cpu().numpy()
    path_len = np.sum(np.linalg.norm(np.diff(preds[:20, :2], axis=0), axis=1))
    dist_fin = np.linalg.norm(preds[-1, :2] - dest_ego)

    print(f"  preds[0]  ego=({preds[0,0]:.2f}, {preds[0,1]:.2f})")
    print(f"  preds[-1] ego=({preds[-1,0]:.2f}, {preds[-1,1]:.2f})")
    print(f"  Path length first 20 wps: {path_len:.3f}m")
    print(f"  Final wp dist to dest:    {dist_fin:.3f}m")

    all_ok  = True
    all_ok &= check(np.all(np.isfinite(preds)), "preds finite")
    all_ok &= check(path_len > 1.0,
                    f"model output is spread (path={path_len:.2f}m > 1m)",
                    "model outputs a dot — all waypoints at same location")
    all_ok &= check(dist_fin < np.linalg.norm(dest_ego),
                    f"guidance pulls final wp toward dest (dist={dist_fin:.2f}m)",
                    "guidance has no effect on final waypoint")
    return all_ok, preds, anchor


# ---------------------------------------------------------------------------
# Stage 2 — ego→world transform preserves spread
# ---------------------------------------------------------------------------

def test_stage2_ego_to_world(preds, anchor):
    print("\n=== Stage 2: ego-centric std → world CARLA ===")

    cos_h = preds[:, 2]; sin_h = preds[:, 3]
    norm  = np.sqrt(cos_h**2 + sin_h**2)
    norm  = np.where(norm < 1e-6, 1., norm)
    local_poses = np.stack([preds[:, 0], preds[:, 1], np.arctan2(sin_h/norm, cos_h/norm)], axis=1)

    anchor_mat   = _state_se2_array_to_transform_matrix(anchor)
    pose_mats    = _state_se2_array_to_transform_matrix_batch(local_poses)
    global_poses = _transform_matrix_to_state_se2_array_batch(anchor_mat @ pose_mats)
    carla_pts    = np.array([standard_to_carla(gx, gy, gh)[:2] for gx, gy, gh in global_poses])

    path_len = np.sum(np.linalg.norm(np.diff(carla_pts[:20], axis=0), axis=1))
    dist_ego = np.linalg.norm(carla_pts[0] - np.array([EGO_CX, EGO_CY]))

    print(f"  carla_pts[0]:  ({carla_pts[0,0]:.2f}, {carla_pts[0,1]:.2f})")
    print(f"  carla_pts[-1]: ({carla_pts[-1,0]:.2f}, {carla_pts[-1,1]:.2f})")
    print(f"  ego CARLA: ({EGO_CX}, {EGO_CY})   dest CARLA: ({DEST_CX}, {DEST_CY})")
    print(f"  Path length first 20 wps (world): {path_len:.3f}m")
    print(f"  Distance of wp[0] from ego:       {dist_ego:.2f}m  (expect < 5m)")

    all_ok  = True
    all_ok &= check(path_len > 1.0,
                    f"world-frame path is spread (path={path_len:.2f}m)",
                    "transform collapses trajectory to a dot")
    all_ok &= check(dist_ego < 10.0,
                    f"wp[0] near ego (dist={dist_ego:.2f}m < 10m)",
                    "wp[0] far from ego — anchor transform wrong")
    return all_ok, carla_pts


# ---------------------------------------------------------------------------
# Stage 3 — latest_pred_route storage (old ego-frame vs new world-frame)
# ---------------------------------------------------------------------------

def test_stage3_pred_route_storage(carla_pts):
    print("\n=== Stage 3: latest_pred_route storage pipeline ===")

    ego_yaw_rad = np.deg2rad(EGO_YAW_DEG)

    # OLD: agent_interface converts to ego, v3.py calls ego_to_world on top
    ego_route = []
    for pt in carla_pts[:20]:
        dx = pt[0] - EGO_CX; dy = pt[1] - EGO_CY
        c, s = np.cos(-ego_yaw_rad), np.sin(-ego_yaw_rad)
        ego_route.append([dx*c - dy*s, dx*s + dy*c])
    world_from_old = _ego_to_world(np.array(ego_route), EGO_CX, EGO_CY, ego_yaw_rad)

    # NEW: world coords stored directly, v3.py uses as-is
    world_from_new = carla_pts[:20].copy()

    path_old      = np.sum(np.linalg.norm(np.diff(world_from_old, axis=0), axis=1))
    path_new      = np.sum(np.linalg.norm(np.diff(world_from_new, axis=0), axis=1))
    pipeline_diff = np.mean(np.linalg.norm(world_from_old - world_from_new, axis=1))

    print(f"  OLD pipeline path (double-transformed): {path_old:.3f}m")
    print(f"  NEW pipeline path (world direct):       {path_new:.3f}m")
    print(f"  Mean diff old vs new:                   {pipeline_diff:.4f}m")
    print(f"  old[0]: ({world_from_old[0,0]:.2f}, {world_from_old[0,1]:.2f})  "
          f"new[0]: ({world_from_new[0,0]:.2f}, {world_from_new[0,1]:.2f})")

    if pipeline_diff < 0.1:
        print("  NOTE: Pipelines agree — double-transform is identity. Bug is NOT here; look at Stage 4.")
    else:
        print(f"  NOTE: Pipelines diverge by {pipeline_diff:.2f}m — double-transform IS the dot bug.")

    all_ok  = True
    all_ok &= check(path_new > 1.0,
                    f"NEW storage gives spread world path (path={path_new:.2f}m)",
                    "world-direct storage collapses to dot")
    return all_ok, world_from_new


# ---------------------------------------------------------------------------
# Stage 4 — project_world_to_topdown produces distinct, in-frame pixels
# ---------------------------------------------------------------------------

def test_stage4_pixel_projection(world_route):
    print("\n=== Stage 4: world coords → top-down pixels ===")

    pixels   = [_project(x, y, EGO_CX, EGO_CY) for x, y in world_route]
    unique   = len(set(pixels))
    range_x  = max(p[0] for p in pixels) - min(p[0] for p in pixels)
    range_y  = max(p[1] for p in pixels) - min(p[1] for p in pixels)
    in_frame = sum(0 <= p[0] < CAM_WIDTH and 0 <= p[1] < CAM_HEIGHT for p in pixels)

    print(f"  Camera {CAM_WIDTH}x{CAM_HEIGHT}, cam_z={CAM_Z}m, "
          f"visible={2*CAM_Z}m, ppm={CAM_WIDTH/(2*CAM_Z):.1f}")
    print(f"  pixels[0]: {pixels[0]}   pixels[-1]: {pixels[-1]}")
    print(f"  Unique positions: {unique}/{len(pixels)}")
    print(f"  Pixel spread: dx={range_x}px  dy={range_y}px")
    print(f"  In-frame: {in_frame}/{len(pixels)}")
    print(f"  All pixels: {pixels}")

    all_ok  = True
    all_ok &= check(unique > 3,
                    f"trajectory maps to multiple pixels (unique={unique})",
                    "all waypoints map to same pixel")
    all_ok &= check(range_x + range_y > 5,
                    f"pixel spread visible (dx={range_x}+dy={range_y} > 5px)",
                    "waypoints too close in pixel space")
    all_ok &= check(in_frame > len(pixels) // 2,
                    f"most waypoints inside frame ({in_frame}/{len(pixels)})",
                    "trajectory outside the camera visible area")
    return all_ok


# ---------------------------------------------------------------------------
# Stage 0 — Production guidance scales (GOAL=200, PATH=100), no agents
# ---------------------------------------------------------------------------

PROD_GOAL_SCALE = 200.0
PROD_PATH_SCALE = 100.0

def test_stage0_production_scales_no_agents(device):
    """
    Run the model with the ACTUAL diff_adapter.py guidance scales (200/100).
    The existing Stage 1 uses 5000/2500 which always passes. This checks whether
    the weaker production scales still produce a spread trajectory, or if the dot
    is caused by under-powered guidance.
    """
    print("\n=== Stage 0: production guidance scales (200/100), no agents ===")

    anchor, dest_ego = compute_dest_ego()
    print(f"  dest in ego frame: ({dest_ego[0]:.2f}, {dest_ego[1]:.2f})  "
          f"dist={np.linalg.norm(dest_ego):.2f}m")
    print(f"  GOAL_SCALE={PROD_GOAL_SCALE}  PATH_SCALE={PROD_PATH_SCALE}  (diff_adapter.py production values)")

    gfn   = _make_guidance_fn(dest_ego, goal_scale=PROD_GOAL_SCALE, path_scale=PROD_PATH_SCALE)
    model, config = _load_model(gfn, device)

    route_lanes = _build_route_lanes(config, dest_ego)
    inputs      = _make_inputs(device, config, route_lanes)
    if config.observation_normalizer:
        inputs = config.observation_normalizer(inputs)

    with torch.no_grad():
        _, outputs = model(inputs)

    preds    = outputs["prediction"][0, 0].cpu().numpy()
    path_len = np.sum(np.linalg.norm(np.diff(preds[:20, :2], axis=0), axis=1))
    dist_fin = np.linalg.norm(preds[-1, :2] - dest_ego)

    print(f"  preds[0]  ego=({preds[0,0]:.3f}, {preds[0,1]:.3f})")
    print(f"  preds[5]  ego=({preds[5,0]:.3f}, {preds[5,1]:.3f})")
    print(f"  preds[-1] ego=({preds[-1,0]:.3f}, {preds[-1,1]:.3f})")
    print(f"  Path length first 20 wps: {path_len:.3f}m")
    print(f"  Final wp dist to dest:    {dist_fin:.3f}m  (dest dist={np.linalg.norm(dest_ego):.2f}m)")

    all_ok  = True
    all_ok &= check(np.all(np.isfinite(preds)), "preds finite (no NaN/Inf)")
    all_ok &= check(path_len > 1.0,
                    f"production scales produce spread traj (path={path_len:.2f}m > 1m)",
                    "DOT BUG: production guidance scales (200/100) are too weak — increase in diff_adapter.py")
    return all_ok


# ---------------------------------------------------------------------------
# Stage 1b — Production scales WITH synthetic agents
# ---------------------------------------------------------------------------

def _build_synthetic_agent_buffer(anchor):
    """
    Build an AgentHistoryBuffer with 6 parked cars mirroring the Town04
    parking lot geometry: 3 to the left (~5m lateral), 3 to the right (~5m lateral),
    each at forward offsets of 5m, 10m, 15m.

    These are PARKED cars (speed=0), NOT blocking the forward path — this
    matches the live CARLA scenario where adjacent-row cars are to the sides.

    anchor: (3,) = [std_x, std_y, std_h] of ego in global standard frame
    """
    anchor_mat = _state_se2_array_to_transform_matrix(anchor)

    buf = AgentHistoryBuffer()
    for _ in range(21):   # fill full 21-step history so no zero-padded frames
        states = []
        actor_id = 0
        for forward_m in [5.0, 10.0, 15.0]:
            for lateral_m in [-5.3, +5.6]:   # right / left (~Row 2 / Row 3 offsets)
                local_h = np.array([forward_m, lateral_m, 1.0])
                global_pos = anchor_mat @ local_h
                states.append(AgentState(
                    actor_id   = actor_id,
                    x          = float(global_pos[0]),
                    y          = float(global_pos[1]),
                    heading    = float(anchor[2]),   # parked, same orientation as ego
                    vx         = 0.0,
                    vy         = 0.0,
                    width      = 2.0,
                    length     = 4.5,
                    agent_type = 'vehicle',
                ))
                actor_id += 1
        buf.update(states)
    return buf


def test_stage1b_with_agents(device):
    """
    Run the model with production guidance scales AND synthetic neighbor agents.

    Sub-test A: 6 parked cars at realistic parking lot positions (the live failure mode).
    Sub-test B: same parked cars but FILTERED OUT (speed=0, below _MIN_AGENT_SPEED_MS=0.5).
               This verifies that filtering stationary agents fixes the dot.
    Sub-test C: 2 moving agents (speed=3 m/s) to confirm moving agents don't break the model.
    """
    print("\n=== Stage 1b: production scales + synthetic agents (parking lot scenario) ===")

    anchor, dest_ego = compute_dest_ego()
    MAX_SPEED = 14.0

    gfn   = _make_guidance_fn(dest_ego, goal_scale=PROD_GOAL_SCALE, path_scale=PROD_PATH_SCALE)
    model, config = _load_model(gfn, device)
    route_lanes = _build_route_lanes(config, dest_ego)

    def _run_with_agents(agents_tensor, label):
        nonzero = np.any(agents_tensor != 0, axis=(1, 2)).sum()
        print(f"  [{label}] non-zero agent slots: {nonzero}/32")
        data = {
            "neighbor_agents_past":        agents_tensor,
            "ego_current_state":           np.array([0.,0.,1.,0.,1.5,0.,0.,0.,0.,0.], dtype=np.float32),
            "static_objects":              np.zeros((5, 10), dtype=np.float32),
            "lanes":                       np.zeros((LANE_NUM, 20, 12), dtype=np.float32),
            "lanes_speed_limit":           np.full((LANE_NUM, 1), MAX_SPEED, dtype=np.float32),
            "lanes_has_speed_limit":       np.ones((LANE_NUM, 1), dtype=np.bool_),
            "route_lanes":                 route_lanes,
            "route_lanes_speed_limit":     np.full((config.route_num, 1), MAX_SPEED, dtype=np.float32),
            "route_lanes_has_speed_limit": np.ones((config.route_num, 1), dtype=np.bool_),
        }
        inp = convert_to_model_inputs(data, device)
        if config.observation_normalizer:
            inp = config.observation_normalizer(inp)
        with torch.no_grad():
            _, out = model(inp)
        p = out["prediction"][0, 0].cpu().numpy()
        path = np.sum(np.linalg.norm(np.diff(p[:20, :2], axis=0), axis=1))
        print(f"  [{label}] preds[0]=({p[0,0]:.2f},{p[0,1]:.2f})  "
              f"preds[-1]=({p[-1,0]:.2f},{p[-1,1]:.2f})  path20={path:.2f}m")
        return p, path

    # ------------------------------------------------------------------
    # Sub-test A: 6 parked cars at realistic positions (live failure mode)
    print("\n  --- Sub-A: 6 parked side cars (speed=0, the old live behavior) ---")
    buf_parked = _build_synthetic_agent_buffer(anchor)
    agents_parked = build_neighbor_agents_past(buf_parked, anchor).astype(np.float32)
    for i in np.where(np.any(agents_parked != 0, axis=(1, 2)))[0]:
        print(f"  Agent[{i}] ego x,y: ({agents_parked[i,-1,0]:.1f}, {agents_parked[i,-1,1]:.1f})")
    preds_a, path_a = _run_with_agents(agents_parked, "parked")

    # ------------------------------------------------------------------
    # Sub-test B: filtered — zero-speed agents excluded (fix in agent_interface.py)
    print("\n  --- Sub-B: zero agents (parked cars filtered out — fixed behavior) ---")
    agents_zero = np.zeros((32, 21, 11), dtype=np.float32)
    preds_b, path_b = _run_with_agents(agents_zero, "filtered")

    # ------------------------------------------------------------------
    # Sub-test C: 2 moving agents (speed=3 m/s) ahead and to the right
    print("\n  --- Sub-C: 2 moving agents (speed=3 m/s, 10m ahead, ±5m lateral) ---")
    buf_moving = AgentHistoryBuffer()
    for step in range(21):
        states = []
        for actor_id, (fwd, lat) in enumerate([(10.0, -5.3), (10.0, 5.6)]):
            local_h = np.array([fwd + step * 0.03, lat, 1.0])   # slowly moving forward
            global_pos = _state_se2_array_to_transform_matrix(anchor) @ local_h
            states.append(AgentState(
                actor_id=actor_id, x=float(global_pos[0]), y=float(global_pos[1]),
                heading=float(anchor[2]), vx=3.0, vy=0.0,
                width=2.0, length=4.5, agent_type='vehicle',
            ))
        buf_moving.update(states)
    agents_moving = build_neighbor_agents_past(buf_moving, anchor).astype(np.float32)
    preds_c, path_c = _run_with_agents(agents_moving, "moving")

    # ------------------------------------------------------------------
    print(f"\n  Summary:")
    print(f"    Parked cars (old):  path20={path_a:.2f}m  {'DOT' if path_a < 1.0 else 'spread'}")
    print(f"    Filtered (fix):     path20={path_b:.2f}m  {'DOT' if path_b < 1.0 else 'spread'}")
    print(f"    Moving agents:      path20={path_c:.2f}m  {'DOT' if path_c < 1.0 else 'spread'}")

    all_ok  = True
    all_ok &= check(np.all(np.isfinite(preds_a)), "sub-A: preds finite (no NaN/Inf)")
    all_ok &= check(path_b > 1.0,
                    f"sub-B: filtered agents give spread traj (path={path_b:.2f}m > 1m)",
                    "fix not working: filtered zero-agent run still collapses")
    all_ok &= check(path_c > 1.0,
                    f"sub-C: moving agents don't collapse traj (path={path_c:.2f}m > 1m)",
                    "moving agents also suppress trajectory — deeper model issue")
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Ego CARLA:  ({EGO_CX}, {EGO_CY}, yaw={EGO_YAW_DEG}°)")
    print(f"Dest CARLA: ({DEST_CX}, {DEST_CY})")

    r0             = test_stage0_production_scales_no_agents(device)
    r1b            = test_stage1b_with_agents(device)
    r1, preds, anchor = test_stage1_model_output_spread(device)
    r2, carla_pts     = test_stage2_ego_to_world(preds, anchor)
    r3, world_route   = test_stage3_pred_route_storage(carla_pts)
    r4                = test_stage4_pixel_projection(world_route)

    print(f"\n{'='*55}")
    passed = sum([r0, r1b, r1, r2, r3, r4])
    print(f"Results: {passed}/6 stages passed")
    if not r0:  print("  → Stage 0 FAIL: production scales (200/100) too weak — increase GOAL_SCALE/PATH_SCALE in diff_adapter.py")
    if not r1b: print("  → Stage 1b FAIL: agents corrupt model output — check build_neighbor_agents_past")
    if not r1:  print("  → Stage 1 FAIL: model dot even with strong guidance (5000/2500) — check route_lanes format")
    if not r2:  print("  → Stage 2 FAIL: transform collapses trajectory — check carla_transform_to_standard / standard_to_carla")
    if not r3:  print("  → Stage 3 FAIL: storage pipeline bug — check agent_interface.py")
    if not r4:  print("  → Stage 4 FAIL: waypoints outside camera frame or collapsed to one pixel")