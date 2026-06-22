"""
test_guidance.py — Verify classifier guidance fires and steers trajectory.

Uses the same import pattern as test_model_smoke.py (no CARLA/srunner needed).
Runs the full DPM sampler with and without guidance to confirm:
  1. Guidance gradient is non-zero (T_GATE bug is fixed)
  2. Guided trajectory differs from unguided (guidance has an effect)
  3. Guided waypoints spread further in the destination direction
  4. No NaN/Inf introduced by guidance

Run with:
    /home/sumesh/envs/simlingo/bin/python test/test_guidance.py
"""

import sys, os
import numpy as np
import torch

_HERE         = os.path.dirname(os.path.abspath(__file__))
_ADAPTER_ROOT = os.path.dirname(_HERE)
_AUTOVALET    = os.path.dirname(_ADAPTER_ROOT)
_NUPLAN_ROOT  = os.path.join(_ADAPTER_ROOT, "nuplan-devkit")
_DP_ROOT      = os.path.join(_ADAPTER_ROOT, "Diffusion-Planner")

# Paths from run_test.sh
_CARLA_ROOT    = "/home/sumesh/opt/carla/PythonAPI/carla"
_SCENARIO_ROOT = "/home/sumesh/carla_garage/scenario_runner"
_LB_ROOT       = "/home/sumesh/carla_garage/leaderboard"

for _p in [_CARLA_ROOT, _SCENARIO_ROOT, _LB_ROOT,
           _HERE, _AUTOVALET, _ADAPTER_ROOT, _NUPLAN_ROOT, _DP_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from diffusion_planner.data_process.utils import (
    convert_to_model_inputs,
    vector_set_coordinates_to_local_frame,
    _state_se2_array_to_transform_matrix,
)
from utils.map_process import LANE_NUM
import numpy.linalg as npl

CKPT = os.path.join(_ADAPTER_ROOT, "checkpoints/model.pth")
ARGS = os.path.join(_ADAPTER_ROOT, "checkpoints/args.json")

# Town04 first parking spot — ego 20m south of dest
EGO_CX,  EGO_CY,  EGO_YAW_DEG = 285.6, -243.7, 90.0
DEST_CX, DEST_CY               = 298.5, -235.73

GOAL_SCALE = 5000.0
PATH_SCALE = 2500.0


def ok(tag):   print(f"  [PASS] {tag}")
def fail(tag, detail=""): print(f"  [FAIL] {tag}  {detail}")
def check(cond, tag, detail=""):
    if cond: ok(tag)
    else:    fail(tag, detail)
    return cond


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

def load_model(device):
    config = Config(ARGS, guidance_fn=None)
    model  = Diffusion_Planner(config)
    raw = torch.load(CKPT, map_location=device)
    sd  = raw.get("ema_state_dict") or raw.get("model") or raw
    sd  = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model.to(device), config


def _build_route_lanes_for_test(config, dest_ego_np, spacing=0.5, half_width=1.75):
    """
    Build route_lanes in the correct nuPlan training format:
      [0:2] ego-centric (x, y) positions
      [2:4] raw step diffs (~spacing metres, NOT unit vectors)
      [4:6] left  boundary offset (half_width × left-perpendicular)
      [6:8] right boundary offset (half_width × right-perpendicular)
      [8:12] traffic light one-hot [0,0,0,1] = unknown

    Ego is at origin; destination is dest_ego_np in ego frame.
    """
    route_num = config.route_num
    route_len = config.route_len

    dest = np.array(dest_ego_np, dtype=np.float64)
    dist = np.linalg.norm(dest)
    if dist < 1e-3:
        return np.zeros((route_num, route_len, 12), dtype=np.float32)

    unit = dest / dist
    total_dist = dist + 15.0
    n = max(2, int(total_dist / spacing) + 1)
    ts = np.linspace(0.0, total_dist, n)
    pts = unit[None, :] * ts[:, None]   # (n, 2) ego-centric

    tl = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (route_len, 1))
    route_lanes = np.zeros((route_num, route_len, 12), dtype=np.float32)

    N = len(pts)
    if N <= route_len:
        n_windows, stride = 1, 0
    else:
        n_windows = min(route_num, N - route_len + 1)
        stride    = max(1, (N - route_len) // max(1, n_windows - 1))

    for i in range(n_windows):
        seg = pts[i * stride : i * stride + route_len]
        if len(seg) < route_len:
            seg = np.vstack([seg, np.tile(seg[-1], (route_len - len(seg), 1))])

        fwd_raw    = np.diff(seg, axis=0)
        fwd_raw    = np.vstack([fwd_raw, fwd_raw[-1:]])
        norms      = np.linalg.norm(fwd_raw, axis=1, keepdims=True)
        safe_norms = np.where(norms < 1e-6, 1.0, norms)
        left_off   = half_width * np.column_stack(
            [-fwd_raw[:, 1],  fwd_raw[:, 0]]) / safe_norms
        right_off  = half_width * np.column_stack(
            [ fwd_raw[:, 1], -fwd_raw[:, 0]]) / safe_norms

        route_lanes[i] = np.concatenate([
            seg.astype(np.float32),
            fwd_raw.astype(np.float32),
            left_off.astype(np.float32),
            right_off.astype(np.float32),
            tl,
        ], axis=1)

    return route_lanes


def make_dummy_inputs(device, config, speed=1.5):
    MAX_SPEED = 14.0
    data = {
        "neighbor_agents_past":         np.zeros((32, 21, 11), dtype=np.float32),
        "ego_current_state":            np.array([0.,0.,1.,0., speed,0.,0.,0.,0.,0.], dtype=np.float32),
        "static_objects":               np.zeros((5, 10), dtype=np.float32),
        "lanes":                        np.zeros((LANE_NUM, 20, 12), dtype=np.float32),
        "lanes_speed_limit":            np.full((LANE_NUM, 1), MAX_SPEED, dtype=np.float32),
        "lanes_has_speed_limit":        np.ones((LANE_NUM, 1), dtype=np.bool_),
        "route_lanes":                  np.zeros((config.route_num, config.route_len, 12), dtype=np.float32),
        "route_lanes_speed_limit":      np.full((config.route_num, 1), MAX_SPEED, dtype=np.float32),
        "route_lanes_has_speed_limit":  np.ones((config.route_num, 1), dtype=np.bool_),
    }
    return convert_to_model_inputs(data, device)


def _compute_anchor_inv():
    # The model's ego_current_state is always at the origin (0,0,heading=0).
    # anchor_inv must match: ego IS the world origin, so anchor_inv = identity.
    # dest_ego is the destination expressed in this ego frame.
    # We place it at a realistic offset from the ego: 10m forward, 5m left.
    anchor_inv = np.eye(3)
    dest_ego   = np.array([10.0, -5.0])   # (2,) ego-centric, matches make_dummy_inputs
    dest_std   = dest_ego.copy()           # world = ego when anchor is identity
    return anchor_inv, dest_ego, dest_std


# ---------------------------------------------------------------------------
# Self-contained guidance function (mirrors diff_adapter._guidance_fn)
# ---------------------------------------------------------------------------

def _make_guidance_fn(state_normalizer, anchor_inv, dest_ego_np, call_log):
    """Returns a guidance_fn closure matching diff_adapter._guidance_fn exactly.

    Energy is computed in REAL ego-centric space (metres), not normalised space.
    This ensures the unit vector toward the destination always has the correct
    sign even when dest_x < normaliser mean (10m).  Gradient magnitude is
    preserved by dividing the scale constants by std_x (≈ 20).
    """
    dest_ego_t = torch.tensor(dest_ego_np, dtype=torch.float32)

    def guidance_fn(x, t, cond, **kwargs):
        call_log.append(float(t.mean().item()) if hasattr(t, "mean") else float(t))

        zero = (x * 0.0).sum()
        model           = kwargs.get("model")
        model_condition = kwargs.get("model_condition")
        if model is None or model_condition is None:
            return zero

        B, P, D = x.shape

        with torch.no_grad():
            x_fix = model(x, t, **model_condition) - x.detach()
        x_fix = x_fix.reshape(B, P, -1, 4)
        x_fix[:, :, 0] = 0.0
        x = x + x_fix.reshape(B, P, -1)

        T_plus1 = D // 4
        future_xy_norm = x[:, 0, :].reshape(B, T_plus1, 4)[:, 1:, :2]

        dest_ego = dest_ego_t.to(x.device)
        mean_xy = state_normalizer.mean[0, 0, :2].to(x.device)
        std_xy  = state_normalizer.std [0, 0, :2].to(x.device)

        # Denormalize to real ego-centric space
        future_xy_real = future_xy_norm * std_xy[None, None, :] + mean_xy[None, None, :]

        # Goal energy in real space
        final_real  = future_xy_real[:, -1, :]
        goal_dist   = torch.norm(final_real - dest_ego[None], dim=-1)
        goal_energy = -(GOAL_SCALE / std_xy[0]) * goal_dist.mean()

        # Path energy in real space
        dest_dist_real = torch.norm(dest_ego)
        if dest_dist_real > 1e-3:
            unit_real        = dest_ego / dest_dist_real.detach()
            forward_progress = (future_xy_real * unit_real[None, None, :]).sum(dim=-1)
            path_energy      = (PATH_SCALE / std_xy[0]) * forward_progress.mean()
        else:
            path_energy = zero

        return goal_energy + path_energy

    return guidance_fn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_guidance_gradient_nonzero(model, config, device):
    """
    Run one guidance call with a plausible x and verify the returned
    gradient w.r.t. x is non-zero. A zero gradient means the T_GATE bug
    (or missing kwargs) is still blocking guidance.
    """
    print("\n=== Test: guidance gradient is non-zero ===")

    call_log = []
    anchor_inv, dest_ego_np, _ = _compute_anchor_inv()
    gfn = _make_guidance_fn(config.state_normalizer, anchor_inv, dest_ego_np, call_log)

    B, P = 1, config.predicted_neighbor_num + 1
    D    = (1 + 80) * 4
    x = torch.randn(B, P, D, device=device, requires_grad=True)

    # t_input for x_start models is marginal_log_mean_coeff — negative
    t = torch.tensor([-0.25], device=device)

    # Build minimal model_condition
    cross_c = torch.zeros(B, P, 192, device=device)
    route   = torch.zeros(B, config.route_num, config.route_len, 4, device=device)
    mask    = torch.ones(B, P - 1, dtype=torch.bool, device=device)

    energy = gfn(
        x, t, None,
        model            = model.decoder.decoder.dit,
        model_condition  = {"cross_c": cross_c, "route_lanes": route,
                            "neighbor_current_mask": mask},
    )

    all_ok = True
    all_ok &= check(torch.isfinite(energy), f"guidance energy finite (got {energy.item():.4f})")

    grad = torch.autograd.grad(energy.sum(), x)[0]
    grad_norm = grad.norm().item()
    print(f"  t_input:      {t.item():.4f}  (log_mean_coeff — always ≤ 0 for x_start)")
    print(f"  energy:       {energy.item():.4f}")
    print(f"  gradient norm: {grad_norm:.6f}")

    all_ok &= check(grad_norm > 1e-6,
                    f"guidance gradient non-zero (norm={grad_norm:.6f})",
                    "gradient=0 → guidance has no effect (T_GATE bug or detach issue)")
    # Gradient should be O(GOAL_SCALE) ≈ 0.2 in normalised space, not O(GOAL_SCALE × std).
    # A norm > 1000 means the energy is computed in real space and multiplies by std=20.
    # Gradient should be O(GOAL_SCALE) in normalised space, not O(GOAL_SCALE × std = × 20).
    # Real-space energy would give norm ≈ GOAL_SCALE × 20; normalised gives ≈ GOAL_SCALE.
    all_ok &= check(grad_norm < GOAL_SCALE * 5,
                    f"gradient not wildly above GOAL_SCALE (norm={grad_norm:.0f})",
                    f"norm >> GOAL_SCALE → energy likely in real space (× std=20 multiplier)")
    return all_ok


def _load_model_with_guidance(gfn, device):
    """Load a fresh model instance with the given guidance_fn wired through Config."""
    config = Config(ARGS, guidance_fn=gfn)
    model  = Diffusion_Planner(config)
    raw = torch.load(CKPT, map_location=device)
    sd  = raw.get("ema_state_dict") or raw.get("model") or raw
    sd  = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model.to(device), config


def test_guided_vs_unguided(config, device):
    """
    Run the full model forward pass twice with the SAME random seed — once
    with guidance, once without. Identical seed → identical xT noise, so any
    difference in the output is purely from guidance steering the DPM steps.
    """
    print("\n=== Test: guided trajectory differs from unguided (same seed) ===")

    call_log = []
    anchor_inv, dest_ego_np, _ = _compute_anchor_inv()
    gfn = _make_guidance_fn(config.state_normalizer, anchor_inv, dest_ego_np, call_log)

    model_no_gfn, config_no = _load_model_with_guidance(None, device)
    model_with_gfn, config_g = _load_model_with_guidance(gfn,  device)

    obs_norm = config_no.observation_normalizer
    inputs = make_dummy_inputs(device, config_no)
    if obs_norm:
        inputs = obs_norm(inputs)

    SEED = 42

    torch.manual_seed(SEED);  torch.cuda.manual_seed(SEED)
    with torch.no_grad():
        _, out_unguided = model_no_gfn(inputs)

    torch.manual_seed(SEED);  torch.cuda.manual_seed(SEED)
    with torch.no_grad():
        _, out_guided = model_with_gfn(inputs)

    pred_u = out_unguided["prediction"][0, 0].cpu().numpy()
    pred_g = out_guided  ["prediction"][0, 0].cpu().numpy()

    diff = np.abs(pred_g - pred_u).mean()
    print(f"  Guidance was called {len(call_log)} times (expect ~10 for 10 DPM steps)")
    print(f"  Mean |guided − unguided| trajectory diff: {diff:.6f}")

    all_ok = True
    all_ok &= check(len(call_log) > 0,
                    f"guidance function was called ({len(call_log)} times)",
                    "guidance never called — check Config(guidance_fn=...) path")
    all_ok &= check(diff > 1e-4,
                    f"guided trajectory differs from unguided (diff={diff:.6f})",
                    "diff≈0 → guidance has no effect on DPM denoising")
    return all_ok


def test_trajectory_spread(config, device):
    """
    With guidance toward the parking destination, the 80-step trajectory
    should span more than 2m along the ego→dest direction. A 'dot' means
    all waypoints cluster in the same location.
    """
    print("\n=== Test: guided trajectory spread toward destination ===")

    call_log = []
    anchor_inv, dest_ego_np, dest_std = _compute_anchor_inv()
    gfn = _make_guidance_fn(config.state_normalizer, anchor_inv, dest_ego_np, call_log)

    model_g, config_g = _load_model_with_guidance(gfn, device)

    obs_norm = config_g.observation_normalizer
    inputs = make_dummy_inputs(device, config_g)
    if obs_norm:
        inputs = obs_norm(inputs)

    with torch.no_grad():
        _, out = model_g(inputs)

    # out["prediction"] is already denormalized ego-centric, shape (B, M, T, 4)
    pred = out["prediction"][0, 0].cpu().numpy()   # (T, 4) ego-centric std frame

    dest_ego = dest_ego_np
    dest_dir = dest_ego / max(np.linalg.norm(dest_ego), 1e-6)

    # Progress of each waypoint along the ego→dest direction
    progress = pred[:, :2] @ dest_dir   # (T,) — positive = toward dest
    total_progress = progress[-1] - progress[0]

    print(f"  Dest in ego frame: ({dest_ego[0]:.2f}, {dest_ego[1]:.2f})  "
          f"dist={np.linalg.norm(dest_ego):.2f}m")
    print(f"  wp[0]  ego-centric: ({pred[0,0]:.2f}, {pred[0,1]:.2f})")
    print(f"  wp[-1] ego-centric: ({pred[-1,0]:.2f}, {pred[-1,1]:.2f})")
    print(f"  Progress along ego→dest: first={progress[0]:.2f}m  last={progress[-1]:.2f}m  "
          f"total={total_progress:.2f}m")

    xy_range = np.max(np.abs(np.diff(pred[:20, :2], axis=0)).sum(axis=1))
    print(f"  Max step-to-step displacement (first 20 wps): {xy_range:.4f}m")

    all_ok = True
    all_ok &= check(np.all(np.isfinite(pred)), "guided prediction is finite (no NaN/Inf)")

    # Trajectory must be spread (no dot): path length of first 20 wps > 1m
    path20 = np.sum(np.linalg.norm(np.diff(pred[:20, :2], axis=0), axis=1))
    print(f"  Path length (first 20 wps): {path20:.2f}m")
    all_ok &= check(path20 > 1.0,
                    f"first 20 waypoints are spread (path={path20:.2f}m > 1m)",
                    "waypoints cluster in one spot — 'dot' bug")

    # Guidance must move the final waypoint closer to dest than no-guidance baseline
    # Load unguided model for comparison
    m_base, c_base = _load_model_with_guidance(None, device)
    inp_base = make_dummy_inputs(device, c_base)
    if c_base.observation_normalizer: inp_base = c_base.observation_normalizer(inp_base)
    torch.manual_seed(42); torch.cuda.manual_seed(42)
    with torch.no_grad():
        _, out_base = m_base(inp_base)
    pred_base = out_base["prediction"][0, 0].cpu().numpy()

    _, dest_ego_np, _ = _compute_anchor_inv()
    dist_guided   = npl.norm(pred[-1, :2]   - dest_ego_np)
    dist_baseline = npl.norm(pred_base[-1, :2] - dest_ego_np)
    print(f"  Dist to dest: guided={dist_guided:.2f}m  unguided={dist_baseline:.2f}m")
    all_ok &= check(dist_guided < dist_baseline,
                    f"guidance moves final wp toward dest ({dist_guided:.2f}m < {dist_baseline:.2f}m)",
                    "guidance moves trajectory AWAY from dest — direction or scale wrong")
    return all_ok


def test_guidance_rightward_destination(config, device):
    """
    Reproduce the live CARLA failure mode: destination at x < normaliser mean
    (10m), requiring large lateral movement.  e.g. dest_ego = (8, -13) mirrors
    a real parking spot at ~7.97m forward and 12.9m right.

    With normalised-space guidance the unit vector has unit_x < 0, which
    penalises forward motion and produces a dot.  With real-space guidance the
    unit vector is correct (+x forward, -y right) and the trajectory spreads.

    This test would have FAILED with the old normalised-space guidance code.
    """
    print("\n=== Test: guidance with rightward dest (x < 10m, mirrors live CARLA) ===")

    # Parking-spot-like destination: 8m forward, 13m to the right
    dest_ego_np = np.array([8.0, -13.0])
    dest_norm_x = (dest_ego_np[0] - 10.0) / 20.0   # = -0.10 — negative!
    print(f"  dest_ego: ({dest_ego_np[0]}, {dest_ego_np[1]})  "
          f"dest_norm_x={dest_norm_x:.3f}  "
          f"(negative → old code pointed guidance BACKWARD)")

    call_log = []
    anchor_inv = np.eye(3)
    gfn = _make_guidance_fn(config.state_normalizer, anchor_inv, dest_ego_np, call_log)
    model_g, config_g = _load_model_with_guidance(gfn, device)

    obs_norm = config_g.observation_normalizer
    inputs = make_dummy_inputs(device, config_g)
    if obs_norm:
        inputs = obs_norm(inputs)

    with torch.no_grad():
        _, out = model_g(inputs)

    pred = out["prediction"][0, 0].cpu().numpy()

    path20 = np.sum(np.linalg.norm(np.diff(pred[:20, :2], axis=0), axis=1))
    print(f"  wp[0]  ego: ({pred[0,0]:.2f}, {pred[0,1]:.2f})")
    print(f"  wp[-1] ego: ({pred[-1,0]:.2f}, {pred[-1,1]:.2f})")
    print(f"  Path length (first 20 wps): {path20:.2f}m")

    dist_to_dest = npl.norm(pred[-1, :2] - dest_ego_np)
    print(f"  Final wp dist to dest: {dist_to_dest:.2f}m  (dest at {dest_ego_np})")

    all_ok = True
    all_ok &= check(np.all(np.isfinite(pred)), "prediction is finite (no NaN/Inf)")
    all_ok &= check(
        path20 > 1.0,
        f"trajectory spread with rightward dest (path={path20:.2f}m > 1m)",
        "dot bug: old normalised-space guidance penalised forward motion when dest_x < 10m",
    )

    # Load unguided baseline for comparison
    m_base, c_base = _load_model_with_guidance(None, device)
    inp_base = make_dummy_inputs(device, c_base)
    if c_base.observation_normalizer: inp_base = c_base.observation_normalizer(inp_base)
    with torch.no_grad():
        _, out_base = m_base(inp_base)
    pred_base = out_base["prediction"][0, 0].cpu().numpy()
    dist_baseline = npl.norm(pred_base[-1, :2] - dest_ego_np)
    print(f"  Dist to dest: guided={dist_to_dest:.2f}m  unguided={dist_baseline:.2f}m")
    all_ok &= check(
        dist_to_dest < dist_baseline,
        f"guidance moves final wp toward rightward dest ({dist_to_dest:.2f}m < {dist_baseline:.2f}m)",
        "guidance fails when dest_x < normaliser mean — unit-vector sign bug",
    )
    return all_ok


def test_trajectory_with_real_route_lanes(config, device):
    """
    Run with realistic route_lanes (correct nuPlan format: raw step diffs +
    boundary offsets).  This test catches the 'dot' bug that all-zero
    route_lanes miss: if fwd vectors are unit-normalised (magnitude 1.0
    instead of ~0.5m), the model sees out-of-distribution input and may
    output a stationary trajectory.

    Checks:
      1. fwd vector magnitude is ~0.5m (not ~1.0 — the unit-vector bug).
      2. Path length of first 20 waypoints > 1m (not a dot).
      3. Final waypoint closer to destination with guidance than without.
    """
    print("\n=== Test: trajectory spread with realistic route_lanes ===")

    call_log = []
    anchor_inv, dest_ego_np, _ = _compute_anchor_inv()
    gfn = _make_guidance_fn(config.state_normalizer, anchor_inv, dest_ego_np, call_log)
    model_g, config_g = _load_model_with_guidance(gfn, device)

    route_lanes = _build_route_lanes_for_test(config_g, dest_ego_np)

    # Verify the format we built before the model even runs
    fwd_mags     = np.linalg.norm(route_lanes[0, :, 2:4], axis=1)
    mean_fwd_mag = fwd_mags[fwd_mags > 1e-6].mean() if (fwd_mags > 1e-6).any() else 0.0
    print(f"  fwd vector mean magnitude: {mean_fwd_mag:.4f}m  (expect 0.3–0.7 at 0.5m spacing)")

    all_ok = True
    all_ok &= check(
        0.3 < mean_fwd_mag < 0.7,
        f"fwd vectors are raw step diffs (~0.5m), not unit vectors (mean={mean_fwd_mag:.3f}m)",
        "magnitude ≈ 1 → unit-vector bug in _make_straight_route_lanes",
    )

    MAX_SPEED = 14.0
    data = {
        "neighbor_agents_past":         np.zeros((32, 21, 11), dtype=np.float32),
        "ego_current_state":            np.array([0.,0.,1.,0., 1.5,0.,0.,0.,0.,0.], dtype=np.float32),
        "static_objects":               np.zeros((5, 10), dtype=np.float32),
        "lanes":                        np.zeros((LANE_NUM, 20, 12), dtype=np.float32),
        "lanes_speed_limit":            np.full((LANE_NUM, 1), MAX_SPEED, dtype=np.float32),
        "lanes_has_speed_limit":        np.ones((LANE_NUM, 1), dtype=np.bool_),
        "route_lanes":                  route_lanes,
        "route_lanes_speed_limit":      np.full((config_g.route_num, 1), MAX_SPEED, dtype=np.float32),
        "route_lanes_has_speed_limit":  np.ones((config_g.route_num, 1), dtype=np.bool_),
    }
    inputs = convert_to_model_inputs(data, device)
    if config_g.observation_normalizer:
        inputs = config_g.observation_normalizer(inputs)

    with torch.no_grad():
        _, out = model_g(inputs)

    pred = out["prediction"][0, 0].cpu().numpy()   # (T, 4) ego-centric

    print(f"  wp[0]  ego: ({pred[0,0]:.2f}, {pred[0,1]:.2f})")
    print(f"  wp[-1] ego: ({pred[-1,0]:.2f}, {pred[-1,1]:.2f})")

    path20 = np.sum(np.linalg.norm(np.diff(pred[:20, :2], axis=0), axis=1))
    print(f"  Path length (first 20 wps): {path20:.2f}m")

    all_ok &= check(np.all(np.isfinite(pred)), "prediction is finite (no NaN/Inf)")
    all_ok &= check(
        path20 > 1.0,
        f"first 20 wps spread with real route_lanes (path={path20:.2f}m > 1m)",
        "dot bug: waypoints cluster in one spot when real route_lanes are used",
    )

    # Guidance should still move final wp toward destination
    m_base, c_base = _load_model_with_guidance(None, device)
    data_base = dict(data)
    data_base["route_lanes"] = _build_route_lanes_for_test(c_base, dest_ego_np)
    inp_base = convert_to_model_inputs(data_base, device)
    if c_base.observation_normalizer:
        inp_base = c_base.observation_normalizer(inp_base)
    torch.manual_seed(42); torch.cuda.manual_seed(42)
    with torch.no_grad():
        _, out_base = m_base(inp_base)
    pred_base = out_base["prediction"][0, 0].cpu().numpy()

    dist_guided   = npl.norm(pred[-1, :2]   - np.array(dest_ego_np))
    dist_baseline = npl.norm(pred_base[-1, :2] - np.array(dest_ego_np))
    print(f"  Dist to dest: guided={dist_guided:.2f}m  unguided={dist_baseline:.2f}m")
    all_ok &= check(
        dist_guided < dist_baseline,
        f"guidance moves final wp toward dest with real route_lanes ({dist_guided:.2f}m < {dist_baseline:.2f}m)",
        "guidance moves trajectory AWAY from dest when real route_lanes are used",
    )
    return all_ok


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model, config = load_model(device)

    r1 = test_guidance_gradient_nonzero(model, config, device)
    r2 = test_guided_vs_unguided(config, device)
    r3 = test_trajectory_spread(config, device)
    r4 = test_guidance_rightward_destination(config, device)
    r5 = test_trajectory_with_real_route_lanes(config, device)

    print(f"\n{'='*50}")
    passed = sum([r1, r2, r3, r4, r5])
    print(f"Results: {passed}/5 tests passed")
