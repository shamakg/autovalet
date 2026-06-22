"""
test_model_live.py — Run the actual model and diagnose what it outputs.

Tests (all with simlingo Python, GPU):
  1. Zeroed route_lanes → what does the model predict? (prior behavior)
  2. Real forward route_lanes → does trajectory go forward?
  3. Right/left destination route_lanes → does trajectory curve correctly?
  4. Measure trajectory displacement magnitude (is it near-zero?)
  5. Does GOAL_SCALE guidance change the output direction?

Run from diffusion_adapter dir:
    /home/sumesh/envs/simlingo/bin/python test/test_model_live.py
"""

import sys, os
import numpy as np
import torch

# ── path setup ─────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.dirname(_HERE)    # diffusion_adapter/
_NUPLAN      = os.path.join(_ROOT, "nuplan-devkit")
_DP          = os.path.join(_ROOT, "Diffusion-Planner")

for _p in [_HERE, _ROOT, _NUPLAN, _DP]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from diffusion_planner.data_process.utils import (
    convert_to_model_inputs,
    _state_se2_array_to_transform_matrix,
    _state_se2_array_to_transform_matrix_batch,
    _transform_matrix_to_state_se2_array_batch,
    vector_set_coordinates_to_local_frame,
)
from utils.coord_utils import carla_transform_to_standard, standard_to_carla

CKPT = os.path.join(_ROOT, "checkpoints", "model.pth")
ARGS = os.path.join(_ROOT, "checkpoints", "args.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── helpers ─────────────────────────────────────────────────────────────────

def PASS(msg): print(f"  [PASS] {msg}")
def FAIL(msg): print(f"  [FAIL] {msg}")
def INFO(msg): print(f"  [INFO] {msg}")

def check(cond, msg, extra=""):
    if cond: PASS(f"{msg}  {extra}")
    else:    FAIL(f"{msg}  {extra}")
    return cond


def load_model():
    print(f"\n=== Loading model (device={DEVICE}) ===")
    config = Config(ARGS, guidance_fn=None)
    model  = Diffusion_Planner(config)
    raw    = torch.load(CKPT, map_location=DEVICE)
    sd     = raw.get("ema_state_dict") or raw.get("model") or raw
    sd     = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval().to(DEVICE)
    PASS("model loaded")
    return model, config


def make_route(anchor, dest_std, config, add_shift=True):
    """Build (route_num, route_len, 12) route_lanes in ego-centric frame."""
    route_num, route_len = config.route_num, config.route_len
    ego_xy = anchor[:2]
    diff   = np.array(dest_std) - ego_xy
    dist   = np.linalg.norm(diff)
    if dist < 1e-3:
        return np.zeros((route_num, route_len, 12), dtype=np.float32)

    unit  = diff / dist
    total = dist + 15.0
    ts    = np.linspace(0.0, total, route_len)
    pts_g = ego_xy[None, :] + unit[None, :] * ts[:, None]

    avails  = np.ones((1, route_len), dtype=np.bool_)
    pts_ego = vector_set_coordinates_to_local_frame(pts_g[np.newaxis], avails, anchor)[0]

    if add_shift:
        pts_ego[:, 0] += 10.0

    fwd = np.diff(pts_ego, axis=0)
    fwd = np.vstack([fwd, fwd[-1:]])
    tl  = np.tile([0, 0, 0, 1], (route_len, 1)).astype(np.float32)
    base = np.concatenate([
        pts_ego.astype(np.float32), fwd.astype(np.float32),
        np.zeros((route_len, 4), np.float32), tl
    ], axis=1)  # (route_len, 12)

    route_lanes = np.zeros((route_num, route_len, 12), np.float32)
    for i in range(route_num):
        route_lanes[i] = base
    return route_lanes


def run_model(model, config, route_lanes, anchor, speed=1.5):
    """Single forward pass → (T,4) ego-centric predictions."""
    from utils.map_process import LANE_NUM, LANE_LEN
    data = {
        "neighbor_agents_past":         np.zeros((32, 21, 11), np.float32),
        "ego_current_state":            np.array([0,0,1,0,speed,0,0,0,0,0], np.float32),
        "static_objects":               np.zeros((5, 10), np.float32),
        "lanes":                        np.zeros((LANE_NUM, LANE_LEN, 12), np.float32),
        "lanes_speed_limit":            np.full((LANE_NUM, 1), 14., np.float32),
        "lanes_has_speed_limit":        np.ones((LANE_NUM, 1), dtype=np.bool_),
        "route_lanes":                  route_lanes,
        "route_lanes_speed_limit":      np.full((config.route_num, 1), 14., np.float32),
        "route_lanes_has_speed_limit":  np.ones((config.route_num, 1), dtype=np.bool_),
    }
    inputs = convert_to_model_inputs(data, DEVICE)
    if config.observation_normalizer:
        inputs = config.observation_normalizer(inputs)

    with torch.no_grad():
        _, out = model(inputs)
    pred = out["prediction"][0, 0].cpu().numpy()  # (T, 4)

    # Normalize heading before arctan2
    c, s = pred[:, 2], pred[:, 3]
    norm = np.sqrt(c**2 + s**2); norm = np.where(norm < 1e-6, 1., norm)
    headings = np.arctan2(s/norm, c/norm)
    local_poses = np.stack([pred[:, 0], pred[:, 1], headings], axis=1)

    # Ego→global
    A = _state_se2_array_to_transform_matrix(anchor)
    P = _state_se2_array_to_transform_matrix_batch(local_poses)
    G = _transform_matrix_to_state_se2_array_batch(A @ P)

    return pred, G   # (T,4) ego, (T,3) global


def ego_displacement_stats(pred):
    """Return distances from ego-origin for each waypoint."""
    return np.linalg.norm(pred[:, :2], axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1 — zero route → what does the model do?
# ═══════════════════════════════════════════════════════════════════════════════

def test_1_zero_route(model, config):
    """
    Feed all-zero route_lanes. This tests the model's pure prior with no route info.
    If the model has a strong circular prior, we'll see it here.
    """
    print("\n=== Test 1: zero route_lanes (pure model prior) ===")

    anchor = np.array([0.0, 0.0, 0.0])   # ego at origin facing +x
    route_zeros = np.zeros((config.route_num, config.route_len, 12), np.float32)

    pred, global_poses = run_model(model, config, route_zeros, anchor)
    dists = ego_displacement_stats(pred)

    T = pred.shape[0]
    INFO(f"Horizon: T={T} steps ({T*0.1:.1f}s)")
    INFO(f"Max ego-centric displacement: {dists.max():.3f} m")
    INFO(f"Final ego-centric pos: ({pred[-1,0]:.3f}, {pred[-1,1]:.3f})")
    INFO(f"Mean y (lateral): {pred[:,1].mean():.3f} m  (>0 = left, <0 = right)")

    all_ok = True
    all_ok &= check(dists.max() > 0.1, "predictions not stuck at origin", f"max={dists.max():.3f}m")
    all_ok &= check(np.all(np.isfinite(pred)), "all predictions finite")

    if dists.max() < 1.0:
        FAIL("MODEL OUTPUTS NEAR-ZERO DISPLACEMENT WITH ZERO ROUTE — model may be ignoring inputs")
    elif dists.max() < 5.0:
        INFO("Small displacement with zero route — model has weak prior motion")
    else:
        INFO(f"Non-trivial displacement with zero route ({dists.max():.1f}m)")

    # Check if trajectory curves (circular motion signature)
    # In a circle, y will oscillate; x will not monotonically increase
    x_vals = pred[:, 0]
    y_vals = pred[:, 1]
    x_mono = np.all(np.diff(x_vals) >= -0.01)  # mostly non-decreasing
    y_range = y_vals.max() - y_vals.min()
    INFO(f"x monotone: {x_mono}  |  y range: {y_range:.3f}m  (large y-range → circular)")
    if y_range > 5.0 and not x_mono:
        INFO(">>> CIRCULAR MOTION SIGNATURE DETECTED in zero-route prior <<<")

    return all_ok, pred


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2 — forward route → does trajectory go forward?
# ═══════════════════════════════════════════════════════════════════════════════

def test_2_forward_route(model, config):
    """
    Feed a route 60m straight ahead. Trajectory should go forward (+x in ego frame).
    """
    print("\n=== Test 2: forward route (60m ahead) ===")

    ego_cx, ego_cy, ego_yaw = 285.6, -243.7, 90.0
    dst_cx, dst_cy           = 285.6, -183.7   # 60m ahead in CARLA +Y

    anchor   = np.array(carla_transform_to_standard(ego_cx, ego_cy, ego_yaw))
    dest_std = np.array(carla_transform_to_standard(dst_cx, dst_cy, 0.)[:2])

    route = make_route(anchor, dest_std, config)
    INFO(f"Route pt[0] ego: ({route[0,0,0]:.2f}, {route[0,0,1]:.2f})")
    INFO(f"Route pt[-1] ego: ({route[0,-1,0]:.2f}, {route[0,-1,1]:.2f})")

    pred, global_poses = run_model(model, config, route, anchor)
    dists = ego_displacement_stats(pred)

    INFO(f"Max ego displacement: {dists.max():.3f} m")
    INFO(f"Final ego pos: ({pred[-1,0]:.3f}, {pred[-1,1]:.3f})")
    INFO(f"Mean ego-x (forward): {pred[:,0].mean():.3f}")
    INFO(f"Mean ego-y (lateral): {pred[:,1].mean():.3f}")

    # Waypoints in CARLA frame
    wp5  = standard_to_carla(*global_poses[5])
    wpL  = standard_to_carla(*global_poses[-1])
    INFO(f"wp[5]  CARLA: ({wp5[0]:.2f}, {wp5[1]:.2f})")
    INFO(f"wp[-1] CARLA: ({wpL[0]:.2f}, {wpL[1]:.2f})")
    INFO(f"Ego   CARLA: ({ego_cx}, {ego_cy})")
    INFO(f"Dest  CARLA: ({dst_cx}, {dst_cy})")

    all_ok = True
    all_ok &= check(pred[:,0].mean() > 1.0,
                    "Forward route → mean ego-x > 1m (trajectory goes forward)",
                    f"got {pred[:,0].mean():.3f}m")
    all_ok &= check(abs(pred[:,1].mean()) < 5.0,
                    "Forward route → |mean ego-y| < 5m (not spinning sideways)",
                    f"got {pred[:,1].mean():.3f}m")
    all_ok &= check(dists.max() > 0.5,
                    "Trajectory has non-trivial displacement",
                    f"max={dists.max():.3f}m")

    return all_ok, pred


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3 — left vs right routes → does the model steer correctly?
# ═══════════════════════════════════════════════════════════════════════════════

def test_3_left_vs_right(model, config):
    """
    Two routes: one 10m to the right of ego, one 10m to the left.
    In ego frame: right = -y, left = +y.
    The model trajectory mean-y should match the route direction.
    """
    print("\n=== Test 3: left vs right destination ===")

    ego_cx, ego_cy, ego_yaw = 285.6, -243.7, 90.0
    anchor = np.array(carla_transform_to_standard(ego_cx, ego_cy, ego_yaw))

    # CARLA yaw=90 → forward=+Y, right=+X, left=-X
    right_dest = np.array(carla_transform_to_standard(295.6, -243.7, 0.)[:2])  # +10m X
    left_dest  = np.array(carla_transform_to_standard(275.6, -243.7, 0.)[:2])  # -10m X

    route_right = make_route(anchor, right_dest, config)
    route_left  = make_route(anchor, left_dest,  config)

    pred_right, _ = run_model(model, config, route_right, anchor)
    pred_left,  _ = run_model(model, config, route_left,  anchor)

    my_right = pred_right[:, 1].mean()
    my_left  = pred_left[:, 1].mean()

    INFO(f"Right dest route:  mean ego-y = {my_right:.3f}  (expect < 0)")
    INFO(f"Left  dest route:  mean ego-y = {my_left:.3f}   (expect > 0)")

    all_ok = True
    all_ok &= check(my_right < my_left,
                    "Right route → more negative ego-y than left route",
                    f"right={my_right:.3f}, left={my_left:.3f}")
    all_ok &= check(my_right < 0,
                    "Right route → mean ego-y < 0",
                    f"got {my_right:.3f}")
    all_ok &= check(my_left > 0,
                    "Left route → mean ego-y > 0",
                    f"got {my_left:.3f}")

    if not all_ok:
        INFO(">>> Model is NOT responding to route direction — ")
        INFO("    route_lanes may be ignored by the model encoder.")

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4 — displacement magnitude: is the output short/near-zero?
# ═══════════════════════════════════════════════════════════════════════════════

def test_4_displacement_magnitude(model, config):
    """
    The 'short path perpendicular to car' symptom could mean:
      a) Model outputs near-zero ego displacement (< 0.5m total) — all waypoints at origin
      b) Model outputs reasonable displacement but perpendicular to car

    This test checks both by looking at x vs y magnitudes.
    """
    print("\n=== Test 4: displacement magnitude and direction breakdown ===")

    ego_cx, ego_cy, ego_yaw = 285.6, -243.7, 90.0
    dst_cx, dst_cy           = 285.6, -183.7

    anchor   = np.array(carla_transform_to_standard(ego_cx, ego_cy, ego_yaw))
    dest_std = np.array(carla_transform_to_standard(dst_cx, dst_cy, 0.)[:2])
    route    = make_route(anchor, dest_std, config)

    pred, _ = run_model(model, config, route, anchor)
    T = pred.shape[0]

    x_rms = np.sqrt((pred[:,0]**2).mean())
    y_rms = np.sqrt((pred[:,1]**2).mean())
    total_rms = np.sqrt(((pred[:,:2]**2).sum(axis=1)).mean())

    print(f"\n  Ego-frame prediction stats:")
    print(f"    x RMS (forward):  {x_rms:.4f} m")
    print(f"    y RMS (lateral):  {y_rms:.4f} m")
    print(f"    total RMS:        {total_rms:.4f} m")
    print(f"    x/y ratio:        {x_rms/(y_rms+1e-6):.2f}  (>>1 = forward, <<1 = sideways)")
    print()
    for i in [0, 1, 4, 9, 19, T-1]:
        if i < T:
            print(f"    pred[{i:2d}]: ego ({pred[i,0]:7.3f}, {pred[i,1]:7.3f})  "
                  f"dist={np.linalg.norm(pred[i,:2]):.3f}m")

    all_ok = True

    if total_rms < 0.1:
        FAIL(f"NEAR-ZERO displacement (RMS={total_rms:.4f}m) — model stuck at origin!")
        INFO("This means the model treats our inputs as a stopped scene.")
        INFO("→ Check if ego_current_state vx or route_lanes are zero after normalization.")
        all_ok = False
    elif total_rms < 1.0:
        INFO(f"Very small displacement (RMS={total_rms:.3f}m) — model barely moves.")
        all_ok &= check(False, "Total RMS > 1m (model should predict meaningful motion)", f"got {total_rms:.3f}m")
    else:
        PASS(f"Non-trivial displacement (RMS={total_rms:.3f}m)")

    xy_ratio = x_rms / (y_rms + 1e-6)
    if xy_ratio < 0.5:
        FAIL(f"x/y ratio = {xy_ratio:.2f} — trajectory is PREDOMINANTLY LATERAL (sideways!)")
        INFO("This is the perpendicular-path symptom.")
        INFO("→ Route lane direction vectors (fwd) may have x/y swapped.")
        all_ok = False
    elif xy_ratio < 1.0:
        INFO(f"x/y ratio = {xy_ratio:.2f} — trajectory leans lateral (some sideways motion)")
    else:
        PASS(f"x/y ratio = {xy_ratio:.2f} — trajectory is predominantly forward")

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5 — with vs without +10 shift: does it matter?
# ═══════════════════════════════════════════════════════════════════════════════

def test_5_shift_comparison(model, config):
    """
    Compare model output with and without the +10 x-shift on route_lanes.
    If the model is sensitive to the shift, it suggests route_lanes are being used.
    If insensitive, route_lanes may be ignored.
    """
    print("\n=== Test 5: +10 shift sensitivity ===")

    ego_cx, ego_cy, ego_yaw = 285.6, -243.7, 90.0
    dst_cx, dst_cy           = 285.6, -183.7

    anchor   = np.array(carla_transform_to_standard(ego_cx, ego_cy, ego_yaw))
    dest_std = np.array(carla_transform_to_standard(dst_cx, dst_cy, 0.)[:2])

    route_with    = make_route(anchor, dest_std, config, add_shift=True)
    route_without = make_route(anchor, dest_std, config, add_shift=False)
    route_zeros   = np.zeros((config.route_num, config.route_len, 12), np.float32)

    pred_with, _    = run_model(model, config, route_with,    anchor)
    pred_without, _ = run_model(model, config, route_without, anchor)
    pred_zero, _    = run_model(model, config, route_zeros,   anchor)

    # Compare final positions
    def final(p): return p[-1, :2]
    def fmt(p): return f"({p[0]:.3f}, {p[1]:.3f})"

    diff_shift   = np.linalg.norm(final(pred_with)    - final(pred_without))
    diff_vs_zero_with    = np.linalg.norm(final(pred_with)    - final(pred_zero))
    diff_vs_zero_without = np.linalg.norm(final(pred_without) - final(pred_zero))

    INFO(f"Final ego pos WITH shift:    {fmt(final(pred_with))}")
    INFO(f"Final ego pos WITHOUT shift: {fmt(final(pred_without))}")
    INFO(f"Final ego pos ZERO route:    {fmt(final(pred_zero))}")
    print()
    INFO(f"Diff (with vs without shift): {diff_shift:.4f}m")
    INFO(f"Diff (with shift vs zero):    {diff_vs_zero_with:.4f}m")
    INFO(f"Diff (without shift vs zero): {diff_vs_zero_without:.4f}m")

    all_ok = True

    if diff_vs_zero_with < 0.01 and diff_vs_zero_without < 0.01:
        FAIL("Model output is IDENTICAL for zero vs non-zero route_lanes!")
        INFO(">>> Route_lanes are being IGNORED by the model encoder. <<<")
        INFO("Possible causes:")
        INFO("  a) route_lanes after normalization are so far from training mean they get zeroed")
        INFO("  b) Model architecture drops route_lanes in some condition")
        INFO("  c) route_lanes_has_speed_limit or availability mask zeroing the input")
        all_ok = False
    elif diff_vs_zero_with < 0.1:
        INFO(f"Very weak route effect (Δ={diff_vs_zero_with:.4f}m) — route barely influences output")
        all_ok &= check(False, "Route changes output by > 0.1m", f"got {diff_vs_zero_with:.4f}m")
    else:
        PASS(f"Route_lanes DO influence model output (Δ={diff_vs_zero_with:.3f}m)")

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  Live Model Diagnostic Test")
    print(f"  Device: {DEVICE}")
    print("=" * 65)

    model, config = load_model()
    INFO(f"route_num={config.route_num}, route_len={config.route_len}")

    r1, pred_zero    = test_1_zero_route(model, config)
    r2, pred_fwd     = test_2_forward_route(model, config)
    r3               = test_3_left_vs_right(model, config)
    r4               = test_4_displacement_magnitude(model, config)
    r5               = test_5_shift_comparison(model, config)

    print(f"\n{'=' * 65}")
    results = [r1, r2, r3, r4, r5]
    names   = ["zero route (prior)",
               "forward route direction",
               "left vs right routing",
               "displacement magnitude",
               "+10 shift sensitivity"]
    for i, (name, ok) in enumerate(zip(names, results)):
        print(f"  {'[PASS]' if ok else '[FAIL]'} Test {i+1}: {name}")

    passed = sum(results)
    print(f"\n{passed}/{len(results)} tests passed")

    print("\n=== DIAGNOSIS ===")
    if not r5:
        print("CRITICAL: Route lanes are IGNORED — model outputs same trajectory regardless of route.")
        print("  → The model is running on its prior only → circular/oscillating motion in CARLA.")
        print("  Fix options:")
        print("    1. Check observation_normalizer is actually applied (config.observation_normalizer)")
        print("    2. Verify route_lanes input tensor is non-zero after normalization")
        print("    3. Increase guidance scale (GOAL_SCALE) drastically in diff_adapter.py")
    elif not r2:
        print("Route lanes reach model but trajectory DOESN'T go forward.")
        print("  → Coordinate convention issue in output parsing or route construction.")
    elif not r3:
        print("Forward route works but LEFT/RIGHT not distinguishable.")
        print("  → Model may need stronger lateral signal (wider spread of route lanes).")
    elif r1 and r2 and r3 and r4 and r5:
        print("All tests pass — model behaves correctly offline.")
        print("  → Bug is in the CARLA integration (agent_interface.py or PID tuning).")
        print("  Check: is the lookahead waypoint actually in front of the car?")
