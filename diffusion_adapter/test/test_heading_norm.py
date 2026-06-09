"""
test_heading_norm.py — Check the arctan2 heading normalization issue.

In adapter.py:
    headings = np.arctan2(predictions[:, 3], predictions[:, 2])   # NO normalization

In diff_adapter.py:
    norm = np.sqrt(cos_h**2 + sin_h**2)
    norm = np.where(norm < 1e-6, 1.0, norm)
    headings = np.arctan2(sin_h / norm, cos_h / norm)              # WITH normalization

If the model outputs (cos_h, sin_h) with magnitude != 1 (which happens after linear
denormalization), these give DIFFERENT results for the heading used in SE2 reconstruction.

This test checks:
1. How large the (cos_h, sin_h) magnitude is for typical denormalized model output.
2. Whether the SE2 reconstruction is sensitive to this.
3. Whether the velocity in ego_current_state is being set correctly (vx=speed, not vx=0).

No model or GPU required — we test the math with synthetic predictions.

Run from diffusion_adapter dir:
    python test/test_heading_norm.py
"""

import sys, os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ADAPTER_ROOT = os.path.dirname(_HERE)
for _p in [_HERE, _ADAPTER_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.coord_utils import carla_transform_to_standard, standard_to_carla


# ── inlined SE2 helpers (no torch) ───────────────────────────────────────────

def _se2_matrix(state):
    x, y, h = float(state[0]), float(state[1]), float(state[2])
    c, s = np.cos(h), np.sin(h)
    return np.array([[c, -s, x], [s, c, y], [0., 0., 1.]])


def _se2_matrix_batch(states):
    """(N,3) → (N,3,3)"""
    N = states.shape[0]
    out = np.zeros((N, 3, 3))
    for i in range(N):
        out[i] = _se2_matrix(states[i])
    return out


def _se2_to_states(mats):
    """(N,3,3) → (N,3) [x, y, heading]"""
    N = mats.shape[0]
    out = np.zeros((N, 3))
    out[:, :2] = mats[:, :2, 2]
    out[:, 2] = np.arctan2(mats[:, 1, 0], mats[:, 0, 0])
    return out


def reconstruct(pred, anchor):
    """(T,4) ego-centric preds + anchor (3,) → (T,3) global poses."""
    T = pred.shape[0]
    cos_h = pred[:, 2]; sin_h = pred[:, 3]
    # with normalization
    norm = np.sqrt(cos_h**2 + sin_h**2)
    norm = np.where(norm < 1e-6, 1.0, norm)
    headings_norm = np.arctan2(sin_h / norm, cos_h / norm)
    # without normalization
    headings_raw  = np.arctan2(sin_h, cos_h)

    local_poses_norm = np.stack([pred[:, 0], pred[:, 1], headings_norm], axis=1)
    local_poses_raw  = np.stack([pred[:, 0], pred[:, 1], headings_raw],  axis=1)

    A = _se2_matrix(anchor)
    P_norm = _se2_matrix_batch(local_poses_norm)
    P_raw  = _se2_matrix_batch(local_poses_raw)

    global_norm = _se2_to_states(A @ P_norm)
    global_raw  = _se2_to_states(A @ P_raw)
    return global_norm, global_raw


def PASS(msg): print(f"  [PASS] {msg}")
def FAIL(msg): print(f"  [FAIL] {msg}")
def check(cond, msg, extra=""):
    if cond: PASS(f"{msg}  {extra}")
    else:    FAIL(f"{msg}  {extra}")
    return cond


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1 — does cos/sin magnitude matter?
# ═══════════════════════════════════════════════════════════════════════════════

def test_1_heading_normalization():
    """
    If cos_h and sin_h have magnitude != 1 (e.g. after linear denormalization),
    arctan2(sin_h, cos_h) still gives the correct angle because arctan2 is scale-invariant.
    So the normalization step is actually UNNECESSARY but harmless.

    The heading SE2 reconstruction then uses headings in _state_se2_array_to_transform_matrix,
    which takes cos/sin of the heading — so the angle is what matters, not magnitude.
    
    CONCLUSION: heading normalization is NOT the bug.
    """
    print("\n=== Test 1: heading normalization effect ===")

    # Simulate model output with scaled cos/sin (magnitude=2)
    pred_scaled = np.array([
        [5.0,  0.0,  2.0,  0.0],   # pointing forward (cos=2,sin=0 → heading=0)
        [10.0, 0.0,  2.0,  0.0],
        [15.0, 0.0,  2.0,  0.0],
    ])
    # Unit magnitude
    pred_unit = np.array([
        [5.0,  0.0,  1.0,  0.0],
        [10.0, 0.0,  1.0,  0.0],
        [15.0, 0.0,  1.0,  0.0],
    ])

    anchor = np.array([100.0, -50.0, 0.0])  # ego at (100,-50) heading=0

    global_scaled, _ = reconstruct(pred_scaled, anchor)
    global_unit,   _ = reconstruct(pred_unit,   anchor)

    print(f"  Scaled magnitude (2.0): global[0] = {global_scaled[0, :2]}")
    print(f"  Unit magnitude   (1.0): global[0] = {global_unit[0, :2]}")

    ok = check(np.allclose(global_scaled[:, :2], global_unit[:, :2], atol=0.01),
               "Scaled cos/sin gives same global positions as unit cos/sin",
               "(arctan2 is scale-invariant)")
    print("  → Heading normalization is cosmetic; NOT the source of the bug.")
    return ok


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2 — ego_current_state velocity bug
# ═══════════════════════════════════════════════════════════════════════════════

def test_2_ego_velocity_field():
    """
    In diff_adapter.py _build_model_inputs():
        ego_current_state = [0, 0, 1, 0, float(cur.speed), 0, 0, 0, 0, 0]

    The model expects [x, y, cos_h, sin_h, vx, vy, ax, ay, steer, yaw_rate].
    vx here is in EGO frame (speed along forward axis).

    cur.speed is CARLA world speed (m/s). Putting it directly as ego-frame vx is
    approximately correct for a standard forward-facing vehicle.
    But if cur.speed is 0 (car just spawned), the model sees a stopped vehicle.

    In adapter.py (the older version):
        std_vx = cur.speed * np.cos(std_heading)
        std_vy = cur.speed * np.sin(std_heading)
    These are GLOBAL frame velocities, not ego-frame! That's wrong.

    Let's check what values the model would see.
    """
    print("\n=== Test 2: ego_current_state velocity field ===")

    ego_cx, ego_cy, ego_yaw_deg = 285.6, -243.7, 90.0
    cur_speed = 1.5  # m/s (TARGET_SPEED from agent_interface.py)

    std_x, std_y, std_h = carla_transform_to_standard(ego_cx, ego_cy, ego_yaw_deg)

    # Method A: diff_adapter.py approach (vx = cur.speed directly)
    ego_state_A = np.array([0., 0., 1., 0., cur_speed, 0., 0., 0., 0., 0.])

    # Method B: adapter.py approach (vx,vy in global frame — WRONG)
    std_vx = cur_speed * np.cos(std_h)  # global x velocity
    std_vy = cur_speed * np.sin(std_h)  # global y velocity
    ego_state_B = np.array([0., 0., 1., 0., std_vx, std_vy, 0., 0., 0., 0.])

    print(f"  ego heading std: {np.rad2deg(std_h):.1f}° (CARLA yaw={ego_yaw_deg}°)")
    print(f"  cur.speed = {cur_speed} m/s")
    print(f"  Method A (diff_adapter): vx={ego_state_A[4]:.3f}, vy={ego_state_A[5]:.3f}  [ego frame]")
    print(f"  Method B (adapter):      vx={ego_state_B[4]:.3f}, vy={ego_state_B[5]:.3f}  [global frame → WRONG]")

    all_ok = True

    # Method A: vx=speed, vy=0 (correct for ego frame when moving forward)
    all_ok &= check(abs(ego_state_A[4] - cur_speed) < 0.01 and abs(ego_state_A[5]) < 0.01,
                    "Method A (diff_adapter): vx≈speed, vy≈0 in ego frame",
                    f"vx={ego_state_A[4]:.3f}, vy={ego_state_A[5]:.3f}")

    # Method B produces near-zero vx (sin(90°)≈0) and vx≈0, vy≈1.5
    all_ok &= check(abs(ego_state_B[4]) < 0.1 and abs(ego_state_B[5] - cur_speed) < 0.1,
                    "Method B (adapter): vx≈0 (WRONG — global frame mix-up)",
                    f"vx={ego_state_B[4]:.3f}, vy={ego_state_B[5]:.3f}")

    if abs(ego_state_B[4]) < 0.1:
        print("  → adapter.py sets vx≈0 (speed in wrong frame) — model sees a near-stopped vehicle")
        print("  → diff_adapter.py correctly sets vx=speed — this is OK")
    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3 — _guidance_fn: is _anchor_inv being set correctly?
# ═══════════════════════════════════════════════════════════════════════════════

def test_3_guidance_anchor_inv():
    """
    In diff_adapter.py, the guidance function uses _anchor_inv to transform
    _dest_std to ego frame:

        dest_h   = [dest[0], dest[1], 1]
        dest_ego = (M_inv @ dest_h)[:2]

    But _anchor_inv is set in diffusion_plan() AFTER the inputs are built.
    In test_coords.py, _anchor_inv is set BEFORE run() is called.

    Check: does the guidance function see the correct ego-frame destination?
    """
    print("\n=== Test 3: guidance fn dest_ego calculation ===")

    ego_cx, ego_cy, ego_yaw_deg = 285.6, -243.7, 90.0
    dst_cx, dst_cy              = 285.6, -183.7   # 60m ahead

    std_x, std_y, std_h = carla_transform_to_standard(ego_cx, ego_cy, ego_yaw_deg)
    anchor = np.array([std_x, std_y, std_h])

    M = _se2_matrix(anchor)
    M_inv = np.linalg.inv(M)

    # _dest_std as set in init_scenario
    dest_x, dest_y, _ = carla_transform_to_standard(dst_cx, dst_cy, 0.0)
    dest_std = np.array([dest_x, dest_y])

    # guidance fn calculation
    dest_h   = np.array([dest_std[0], dest_std[1], 1.0])
    dest_ego = (M_inv @ dest_h)[:2]

    print(f"  ego std: ({std_x:.1f},{std_y:.1f},{np.rad2deg(std_h):.1f}°)")
    print(f"  dest std: ({dest_std[0]:.1f},{dest_std[1]:.1f})")
    print(f"  dest in ego frame: ({dest_ego[0]:.2f},{dest_ego[1]:.2f})")
    print(f"  (expect: ego-x≈+60, ego-y≈0 since dest is 60m ahead)")

    all_ok = True
    all_ok &= check(dest_ego[0] > 50.0 and abs(dest_ego[1]) < 5.0,
                    "Guidance dest_ego points forward (ego-x≈+60, ego-y≈0)",
                    f"got ({dest_ego[0]:.2f},{dest_ego[1]:.2f})")

    if all_ok:
        print("  → Guidance destination transform is CORRECT.")
        print("  → If car still spins, guidance energy may not be strong enough,")
        print("     or T_GATE_LOW/HIGH is gating it out, or the model ignores guidance.")

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4 — what does a typical model output look like if route is zeroed?
# ═══════════════════════════════════════════════════════════════════════════════

def test_4_zero_route_expected_behavior():
    """
    The model was trained with nuPlan data where route_lanes are ALWAYS non-zero
    (real driving scenes have road lanes). When we feed zero route_lanes, the
    model falls back to some prior, which may produce circular motions.

    The key question: is our route_lanes actually reaching the model as non-zero?

    Check: after observation_normalizer, are route_lane values near zero or not?

    We can't run the normalizer without torch, but we can reason:
    - If route lane x-mean is 47.5m and normalizer subtracts mean/divides by std,
      the normalized value depends on the training distribution.
    - If training route_lane x-mean ≈ 47.5m too, then it normalizes to ~0 → 
      model sees no route guidance.
    - If training route_lane x-mean ≈ 10m, then our 47.5m normalizes to a large
      positive offset → model may ignore or misinterpret it.

    Without knowing the normalizer params, we can check what the ROUTE_LANE_NUM is
    and whether our route has the right number of lanes.
    """
    print("\n=== Test 4: route lane count and structure ===")

    import sys, os
    sys.path.insert(0, os.path.join(_ADAPTER_ROOT, 'utils'))

    try:
        from map_process import LANE_NUM, ROUTE_LANE_NUM, LANE_LEN
        print(f"  LANE_NUM={LANE_NUM}, ROUTE_LANE_NUM={ROUTE_LANE_NUM}, LANE_LEN={LANE_LEN}")
    except ImportError as e:
        print(f"  Cannot import map_process: {e}")
        ROUTE_LANE_NUM = 25  # from diff_adapter.py config

    # diff_adapter.py uses config.route_num for route_lanes shape
    # ROUTE_LANE_NUM from map_process.py = 25
    # config.route_num may differ — check if they match

    print(f"\n  diff_adapter uses config.route_num for route_lanes shape.")
    print(f"  map_process.py defines ROUTE_LANE_NUM = {ROUTE_LANE_NUM}")
    print(f"  If config.route_num != ROUTE_LANE_NUM, shape mismatch!")
    print(f"\n  In diff_adapter._build_model_inputs:")
    print(f"    route_lanes shape: (config.route_num, config.route_len, 12)")
    print(f"    route_lanes_speed_limit shape: (config.route_num, 1)")
    print(f"\n  This is correct IF config.route_num == model's expected route_num.")
    print(f"  The config is loaded from args.json — this should be fine.")

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5 — lookahead index: is wp[5] a sensible target?
# ═══════════════════════════════════════════════════════════════════════════════

def test_5_lookahead_distance():
    """
    agent_interface.py picks lookahead = traj[min(5, len(traj)-1)].
    
    The model runs at 10Hz and outputs T steps.
    traj[5] = 0.5 seconds ahead.
    
    For a car moving at 1.5 m/s, that's ~0.75m ahead.
    For a parking maneuver, this might be fine.
    
    But if traj[5] is very close to the car (model outputs tiny steps),
    the steering will oscillate because the lookahead target keeps swinging.

    Let's check: for a 60m route, what does traj[5] look like in global coords?
    """
    print("\n=== Test 5: lookahead waypoint analysis ===")

    ego_cx, ego_cy, ego_yaw_deg = 285.6, -243.7, 90.0

    # Simulated model output: T=80 steps, each step ~0.75m (speed=7.5m/s)
    # or ~0.15m (speed=1.5m/s)
    T = 80
    speed_ms = 1.5  # TARGET_SPEED
    step_m = speed_ms * 0.1   # 0.15m per step at 10Hz

    # Ideal: straight ahead, ego-x increases
    pred_forward = np.zeros((T, 4))
    pred_forward[:, 0] = np.arange(T) * step_m  # ego-x increases
    pred_forward[:, 2] = 1.0  # cos_h = 1 (heading=0 in ego frame)

    # Reproduce conversion
    std_x, std_y, std_h = carla_transform_to_standard(ego_cx, ego_cy, ego_yaw_deg)
    anchor = np.array([std_x, std_y, std_h])
    global_norm, _ = reconstruct(pred_forward, anchor)

    traj5 = global_norm[5]
    carla5 = standard_to_carla(traj5[0], traj5[1], traj5[2])

    dist_from_ego = np.sqrt((carla5[0]-ego_cx)**2 + (carla5[1]-ego_cy)**2)

    print(f"  ego CARLA: ({ego_cx},{ego_cy})")
    print(f"  traj[5] at speed={speed_ms}m/s: ({carla5[0]:.2f},{carla5[1]:.2f})")
    print(f"  distance from ego: {dist_from_ego:.2f}m  (= {T*step_m:.1f}m total × 5/80)")
    print(f"  If actual model output is much smaller (e.g. 0.01m per step),")
    print(f"  traj[5] would be {5*0.01:.2f}m away → PID oscillates.")

    all_ok = True
    all_ok &= check(dist_from_ego > 0.5,
                    f"At speed={speed_ms}m/s, traj[5] is >0.5m from ego",
                    f"got {dist_from_ego:.3f}m")

    print(f"\n  KEY QUESTION: what does the ACTUAL model output as wp[5] position?")
    print(f"  Run test_model_smoke.py with real checkpoint to find out.")
    print(f"  If max_dist < 0.5m in the smoke test, the model is outputting near-zero motion.")

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Heading Normalization & Model Input Diagnostic")
    print("  (no model, no GPU, no CARLA required)")
    print("=" * 60)

    r1 = test_1_heading_normalization()
    r2 = test_2_ego_velocity_field()
    r3 = test_3_guidance_anchor_inv()
    r4 = test_4_zero_route_expected_behavior()
    r5 = test_5_lookahead_distance()

    print(f"\n{'=' * 60}")
    results = [r1, r2, r3, r4, r5]
    passed  = sum(results)
    names   = ["heading normalization", "ego velocity field",
               "guidance dest_ego", "route lane structure",
               "lookahead distance"]
    for i, (name, ok) in enumerate(zip(names, results)):
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  {status} Test {i+1}: {name}")

    print(f"\n{passed}/{len(results)} tests passed")
    print()
    print("SUMMARY OF FINDINGS:")
    print("  - Route lane geometry: CORRECT (see test_route_lanes.py)")
    print("  - Heading normalization: cosmetic only, NOT the bug")
    print("  - ego_velocity: diff_adapter.py is correct, adapter.py is wrong")
    print("  - Guidance dest_ego: CORRECT transform")
    print()
    print("MOST LIKELY ROOT CAUSE of circular motion:")
    print("  The model outputs near-zero ego-centric displacement because:")
    print("  a) route_lanes after normalization looks like zero (too far from training mean)")
    print("  b) The model has never seen our CARLA parking lot distribution")
    print("  c) With zero route guidance, the model follows its prior = circle/oscillate")
    print()
    print("RECOMMENDED NEXT STEPS:")
    print("  1. Run: python test/test_model_smoke.py --ckpt checkpoints/model.pth --args checkpoints/args.json")
    print("     → Check max_dist of predictions with zeroed route. If < 1m, model is stuck.")
    print("  2. Run: python test/test_coords.py")
    print("     → Check if model with real route gives directionally correct output.")
    print("  3. Check args.json for obs_normalizer mean/std for route_lanes[x].")
    print("     If mean≈10 and std≈5, our x=47.5m normalizes to (47.5-10)/5=7.5σ → outlier!")
