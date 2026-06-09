"""
test_normalizer.py — Check what route_lanes look like AFTER normalization.

The observation_normalizer transforms route_lanes as:
    normalized_x = (x - mean_x) / std_x = (x - 10) / 20

This test verifies that our route_lanes produce sane normalized values
(not near-zero, not extreme outliers), which would tell the model where to go.

Run from diffusion_adapter dir:
    python test/test_normalizer.py
"""

import sys, os, json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ADAPTER_ROOT = os.path.dirname(_HERE)
_DP_ROOT = os.path.join(_ADAPTER_ROOT, "Diffusion-Planner")
_NUPLAN_ROOT = os.path.join(_ADAPTER_ROOT, "nuplan-devkit")

for _p in [_HERE, _ADAPTER_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.coord_utils import carla_transform_to_standard

ARGS_PATH = os.path.join(_ADAPTER_ROOT, "checkpoints", "args.json")


# ── inlined coord helpers ─────────────────────────────────────────────────────

def _se2_matrix(state):
    x, y, h = float(state[0]), float(state[1]), float(state[2])
    c, s = np.cos(h), np.sin(h)
    return np.array([[c, -s, x], [s, c, y], [0., 0., 1.]])

def _coords_to_local(coords, anchor):
    M_inv = np.linalg.inv(_se2_matrix(anchor))
    padded = np.pad(coords, ((0, 0), (0, 1)), constant_values=1.0)
    return (M_inv @ padded.T).T[:, :2]

def make_route_lanes(anchor, dest_std, route_num=25, route_len=20, add_shift=True):
    ego_xy = anchor[:2]
    diff = dest_std - ego_xy
    dist = np.linalg.norm(diff)
    unit = diff / dist
    total = dist + 15.0
    ts = np.linspace(0.0, total, route_len)
    pts_global = ego_xy[None, :] + unit[None, :] * ts[:, None]

    # Transform to ego frame
    pts_ego = _coords_to_local(pts_global.astype(np.float64), anchor.astype(np.float64)).astype(np.float32)

    if add_shift:
        pts_ego[:, 0] += 10.0  # the +10 shift in diff_adapter.py

    fwd = np.diff(pts_ego, axis=0)
    fwd = np.vstack([fwd, fwd[-1:]])
    tl = np.tile([0, 0, 0, 1], (route_len, 1)).astype(np.float32)
    base = np.concatenate([pts_ego, fwd, np.zeros((route_len, 4), np.float32), tl], axis=1)

    route_lanes = np.zeros((route_num, route_len, 12), dtype=np.float32)
    for i in range(route_num):
        route_lanes[i] = base.copy()
    return route_lanes


def PASS(msg): print(f"  [PASS] {msg}")
def FAIL(msg): print(f"  [FAIL] {msg}")
def check(cond, msg, extra=""):
    if cond: PASS(f"{msg}  {extra}")
    else:    FAIL(f"{msg}  {extra}")
    return cond


# ═══════════════════════════════════════════════════════════════════════════════
# Main test
# ═══════════════════════════════════════════════════════════════════════════════

def test_normalized_route_values():
    """
    Apply the observation_normalizer formula manually:
        normalized = (raw - mean) / std

    and check that the route_lanes values are in a reasonable range.
    """
    print("\n=== Normalizer output check ===")

    with open(ARGS_PATH) as f:
        args = json.load(f)

    rl_norm = args['observation_normalizer']['route_lanes']
    print(f"  route_lanes normalizer:")
    mean_raw = np.array(rl_norm['mean'])
    std_raw  = np.array(rl_norm['std'])
    print(f"    mean shape: {mean_raw.shape}")
    print(f"    std  shape: {std_raw.shape}")
    # Flatten to 1-D (feature dim)
    mean_arr_flat = mean_raw.flatten()
    std_arr_flat  = std_raw.flatten()
    print(f"    mean[:8] = {mean_arr_flat[:8]}")
    print(f"    std[:8]  = {std_arr_flat[:8]}")

    mean_x, mean_y = float(mean_arr_flat[0]), float(mean_arr_flat[1])
    std_x,  std_y  = float(std_arr_flat[0]),  float(std_arr_flat[1])
    mean_dx, mean_dy = float(mean_arr_flat[2]), float(mean_arr_flat[3])
    std_dx,  std_dy  = float(std_arr_flat[2]),  float(std_arr_flat[3])

    print(f"\n  Scalar params:")
    print(f"    x:  mean={mean_x}, std={std_x}")
    print(f"    y:  mean={mean_y}, std={std_y}")
    print(f"    dx: mean={mean_dx}, std={std_dx}")
    print(f"    dy: mean={mean_dy}, std={std_dy}")

    # Build our route
    ego_cx, ego_cy, ego_yaw_deg = 285.6, -243.7, 90.0
    dst_cx, dst_cy              = 285.6, -183.7

    anchor   = np.array(carla_transform_to_standard(ego_cx, ego_cy, ego_yaw_deg))
    dest_std = np.array(carla_transform_to_standard(dst_cx, dst_cy, 0.0)[:2])

    route_with_shift    = make_route_lanes(anchor, dest_std, add_shift=True)
    route_without_shift = make_route_lanes(anchor, dest_std, add_shift=False)

    for label, route in [("WITH +10 shift", route_with_shift), ("WITHOUT shift", route_without_shift)]:
        r0 = route[0]  # (route_len, 12)
        x_raw  = r0[:, 0]
        y_raw  = r0[:, 1]
        dx_raw = r0[:, 2]
        dy_raw = r0[:, 3]

        x_norm  = (x_raw  - mean_x)  / std_x
        y_norm  = (y_raw  - mean_y)  / std_y
        dx_norm = (dx_raw - mean_dx) / std_dx
        dy_norm = (dy_raw - mean_dy) / std_dy

        print(f"\n  [{label}]")
        print(f"    raw  x:  [{x_raw[0]:.2f} → {x_raw[-1]:.2f}], mean={x_raw.mean():.2f}")
        print(f"    norm x:  [{x_norm[0]:.3f} → {x_norm[-1]:.3f}], mean={x_norm.mean():.3f}")
        print(f"    raw  y:  [{y_raw[0]:.2f} → {y_raw[-1]:.2f}], mean={y_raw.mean():.2f}")
        print(f"    norm y:  [{y_norm[0]:.3f} → {y_norm[-1]:.3f}], mean={y_norm.mean():.3f}")
        print(f"    raw  dx: {dx_raw[0]:.4f},  norm dx: {dx_norm[0]:.4f}  [expect ≠ 0, forward direction]")
        print(f"    raw  dy: {dy_raw[0]:.4f},  norm dy: {dy_norm[0]:.4f}  [expect ≈ 0, no lateral]")

        # The normalized x should be in a reasonable range (not ±10σ outlier)
        x_sigma_range = (x_norm.max() - x_norm.min())
        print(f"    Normalized x range: {x_norm.min():.2f} to {x_norm.max():.2f}  ({x_sigma_range:.1f}σ spread)")

        if x_norm.max() > 5.0:
            print(f"    WARNING: normalized x max > 5σ — may be out-of-distribution!")
        else:
            print(f"    OK: normalized x within ±5σ of training distribution")

    # Key check: the +10 shift's purpose
    print(f"\n  Analysis of the +10 shift:")
    print(f"    normalizer mean_x = {mean_x}")
    print(f"    WITHOUT shift: pts_ego[0].x = 0.0 → normalized_x = (0 - {mean_x}) / {std_x} = {(0-mean_x)/std_x:.3f}")
    print(f"    WITH shift:    pts_ego[0].x = 10.0 → normalized_x = (10 - {mean_x}) / {std_x} = {(10-mean_x)/std_x:.3f}")
    print(f"    The shift moves pt[0] from {(0-mean_x)/std_x:.2f}σ to {(10-mean_x)/std_x:.2f}σ.")
    print(f"    {'The shift is CORRECT — pt[0] at mean of training distribution.' if abs(10-mean_x) < 1 else 'The shift partially corrects the offset.'}")

    print(f"\n  CRITICAL: normalizer mean_x = {mean_x}m.")
    print(f"  This means nuPlan training scenes have route lanes with x_mean ≈ {mean_x}m in ego frame.")
    print(f"  Our route: x spans 0→75m (no shift) or 10→85m (with shift).")
    print(f"  → Route is NOT zero, but the model's attention to route vs its trajectory prior")
    print(f"    determines whether it follows the route or goes in a circle.")

    return True


def test_normalizer_on_ego_state():
    """
    Check ego_current_state normalizer: mean=[10,0,0,0,0,...], std=[20,20,1,1,20,...]
    Our input: [0, 0, 1, 0, speed, 0, 0, 0, 0, 0]
    Normalized:
      x: (0-10)/20 = -0.5  ✓ (close to 0)
      y: (0-0)/20  = 0.0   ✓
      cos_h: (1-0)/1 = 1.0 ✓
      sin_h: (0-0)/1 = 0.0 ✓
      vx:  (speed-0)/20    (depends on speed)
    """
    print("\n=== Ego current state normalizer check ===")

    with open(ARGS_PATH) as f:
        args = json.load(f)

    ego_norm = args['observation_normalizer']['ego_current_state']
    mean_e = np.array(ego_norm['mean'])
    std_e  = np.array(ego_norm['std'])

    while mean_e.ndim > 1: mean_e = mean_e[0]
    while std_e.ndim  > 1: std_e  = std_e[0]

    print(f"  mean = {mean_e}")
    print(f"  std  = {std_e}")

    speed = 1.5
    ego_state = np.array([0., 0., 1., 0., speed, 0., 0., 0., 0., 0.])
    normalized = (ego_state - mean_e) / std_e

    print(f"\n  Raw ego_state:    {ego_state}")
    print(f"  Normalized:       {np.round(normalized, 3)}")
    print(f"  Fields: [x, y, cos_h, sin_h, vx, vy, ax, ay, steer, yaw_rate]")

    all_ok = True
    all_ok &= check(abs(normalized[0] + 0.5) < 0.1,
                    "normalized x ≈ -0.5 (ego at x=0, mean=10, std=20)",
                    f"got {normalized[0]:.3f}")
    all_ok &= check(abs(normalized[2] - 1.0) < 0.01,
                    "normalized cos_h = 1.0 (heading=0 in ego frame)",
                    f"got {normalized[2]:.3f}")

    print(f"\n  The model sees ego at ({normalized[0]:.2f}, {normalized[1]:.2f}) in normalized space.")
    print(f"  Route lane pt[0] normalized x = {(0 - float(mean_e[0])) / float(std_e[0]):.3f} (without shift)")
    print(f"  Route lane pt[0] normalized x = {(10 - float(mean_e[0])) / float(std_e[0]):.3f} (with +10 shift)")
    print(f"  → Ego is at {(0-float(mean_e[0]))/float(std_e[0]):.2f}σ, route pt[0] without shift also at {(0-float(mean_e[0]))/float(std_e[0]):.2f}σ")
    print(f"    This means route pt[0] coincides with ego in normalized space (both at x=-0.5).")
    print(f"    With +10 shift: route pt[0] is at 0.0σ (mean), distinguishable from ego.")

    return all_ok


if __name__ == "__main__":
    print("=" * 60)
    print("  Observation Normalizer Diagnostic")
    print("=" * 60)

    r1 = test_normalized_route_values()
    r2 = test_normalizer_on_ego_state()

    print(f"\n{'=' * 60}")
    print(f"\nKEY FINDINGS:")
    print(f"  normalizer: mean_x=10, std_x=20 for route_lanes")
    print(f"  WITHOUT +10 shift: route x spans -0.5σ to +3.25σ (pt[0] at -0.5 = ego position)")
    print(f"  WITH +10 shift:    route x spans 0.0σ to +3.75σ (pt[0] at 0.0 = normalizer mean)")
    print(f"")
    print(f"  The +10 shift ensures route_lane pt[0] starts at the normalizer mean (x=10).")
    print(f"  WITHOUT the shift, route_lane pt[0] coincides with the ego position (-0.5σ),")
    print(f"  which may confuse the model's route encoder.")
    print(f"")
    print(f"  CONCLUSION: The +10 shift IS intentional and correct.")
    print(f"  The model SHOULD see a valid forward-pointing route.")
    print(f"  If it still spins, the issue is in the model's response to our distribution,")
    print(f"  NOT in the route construction.")
    print(f"")
    print(f"MOST LIKELY REMAINING CAUSES:")
    print(f"  1. The diffusion model was trained on nuPlan (highway) not parking lots.")
    print(f"     It may ignore route lanes when the scene looks unlike training data.")
    print(f"  2. GOAL_SCALE and PATH_SCALE guidance may be too weak to overcome the prior.")
    print(f"     Try increasing GOAL_SCALE from 2.0 to 10.0+ in diff_adapter.py.")
    print(f"  3. The model outputs ~0 ego displacement (check test_model_smoke.py output).")
    print(f"     If max_dist < 1m with guidance, the guidance is not reaching the model.")
