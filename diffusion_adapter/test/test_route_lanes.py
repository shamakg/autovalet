"""
test_route_lanes.py  ─  Isolate the circular-spinning bug.

The car spins because the route_lanes tensor fed to the model is WRONG:
either the points don't point forward, the direction vector (fwd) is
perpendicular, or the +10 x-shift breaks the intended direction.

This file contains tightly-scoped tests — no model, no GPU, no CARLA.
Coord-transform functions are inlined to avoid the torch import chain.

Run from the diffusion_adapter directory:
    cd /home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/diffusion_adapter
    python test/test_route_lanes.py
"""

import sys, os
import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_HERE          = os.path.dirname(os.path.abspath(__file__))
_ADAPTER_ROOT  = os.path.dirname(_HERE)          # …/diffusion_adapter/

for _p in [_HERE, _ADAPTER_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.coord_utils import carla_transform_to_standard

# ── inline coord-transform helpers (from diffusion_planner.data_process.utils) ─
# Inlined to avoid pulling in torch which fails in this environment.

def _state_se2_to_matrix(state):  # (3,) → (3,3)
    x, y, h = float(state[0]), float(state[1]), float(state[2])
    c, s = np.cos(h), np.sin(h)
    return np.array([[c, -s, x], [s, c, y], [0., 0., 1.]])


def _coordinates_to_local_frame(coords, anchor_state):
    """(N,2) global → (N,2) ego-centric. Mirrors the diffusion_planner version."""
    M_inv = np.linalg.inv(_state_se2_to_matrix(anchor_state))
    padded = np.pad(coords, ((0, 0), (0, 1)), constant_values=1.0)  # (N,3)
    result = (M_inv @ padded.T).T  # (N,3)
    return result[:, :2]


def vector_set_coordinates_to_local_frame(coords, avails, anchor_state):
    """(num_elements, num_pts, 2) → same shape in ego frame."""
    ne, np_, _ = coords.shape
    flat = coords.reshape(ne * np_, 2)
    flat_local = _coordinates_to_local_frame(flat.astype(np.float64),
                                              anchor_state.astype(np.float64))
    out = flat_local.reshape(ne, np_, 2).astype(np.float32)
    out[~avails] = 0.0
    return out



# ── helpers ───────────────────────────────────────────────────────────────────

def PASS(msg): print(f"  [PASS] {msg}")
def FAIL(msg): print(f"  [FAIL] {msg}")

def check(cond, msg, extra=""):
    if cond: PASS(f"{msg}  {extra}")
    else:    FAIL(f"{msg}  {extra}")
    return cond


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1 ─ coordinate frame: forward is +x in ego frame
# ═══════════════════════════════════════════════════════════════════════════════

def test_1_ego_frame_convention():
    """
    Verify forward/left/right mapping in the ego-centric frame used by nuPlan.
    Forward = +ego_x.  Left = +ego_y.  Right = -ego_y.
    """
    print("\n=== Test 1: ego-frame convention (forward / left / right) ===")

    avails = np.ones((1, 1), dtype=np.bool_)

    def ego_local(global_xy, anchor):
        pt = np.array([[global_xy]], dtype=np.float64)  # (1,1,2)
        return vector_set_coordinates_to_local_frame(pt, avails, anchor)[0, 0]

    # ── heading = 0 (facing +x) ───────────────────────────────────────────────
    anchor_h0 = np.array([0.0, 0.0, 0.0])
    print("\n  Ego at origin facing heading=0 (forward = global +x).")

    fwd  = ego_local([10.0,  0.0], anchor_h0)   # → ego (+10, 0)
    left = ego_local([ 0.0, 10.0], anchor_h0)   # → ego (  0,+10)
    back = ego_local([-10.0, 0.0], anchor_h0)   # → ego (-10, 0)

    print(f"  global( 10,  0) → ego {fwd}   [expect ~(+10,  0)]")
    print(f"  global(  0, 10) → ego {left}  [expect ~(  0,+10)]")
    print(f"  global(-10,  0) → ego {back}  [expect ~(-10,  0)]")

    ok = True
    ok &= check(abs(fwd[0]  - 10.0) < 0.01 and abs(fwd[1]) < 0.01,
                "h=0: forward (+x global) → ego-x ≈ +10, ego-y ≈ 0",
                f"got {fwd}")
    ok &= check(abs(back[0] + 10.0) < 0.01 and abs(back[1]) < 0.01,
                "h=0: backward (-x global) → ego-x ≈ -10, ego-y ≈ 0",
                f"got {back}")

    # ── heading = π/2 (facing +y) ─────────────────────────────────────────────
    anchor_h90 = np.array([0.0, 0.0, np.pi / 2])
    print("\n  Ego at origin facing heading=π/2 (forward = global +y).")

    fwd90  = ego_local([ 0.0, 10.0], anchor_h90)   # → ego (+10, 0)
    right90 = ego_local([10.0,  0.0], anchor_h90)   # → ego (  0,-10)

    print(f"  global(  0, 10) → ego {fwd90}   [expect ~(+10,  0)]")
    print(f"  global( 10,  0) → ego {right90} [expect ~(  0,-10)]")

    ok &= check(abs(fwd90[0] - 10.0) < 0.01 and abs(fwd90[1]) < 0.01,
                "h=π/2: forward (global +y) → ego-x ≈ +10",
                f"got {fwd90}")
    ok &= check(right90[0] < 0.01 and abs(right90[1] + 10.0) < 0.01,
                "h=π/2: right (global +x) → ego-y ≈ -10",
                f"got {right90}")

    return ok


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2 ─ _make_straight_route_lanes produces forward-pointing pts
# ═══════════════════════════════════════════════════════════════════════════════

def _make_straight_route_lanes_no_shift(anchor, dest_std, route_num=25, route_len=20):
    """
    Re-implementation of diff_adapter._make_straight_route_lanes WITHOUT the +10 shift,
    so we can compare raw vs shifted.
    """
    overshoot_m = 15.0
    ego_xy = anchor[:2]

    diff   = dest_std - ego_xy
    dist   = np.linalg.norm(diff)
    if dist < 1e-3:
        return np.zeros((route_num, route_len, 12), dtype=np.float32)

    unit  = diff / dist
    total = dist + overshoot_m
    ts    = np.linspace(0.0, total, route_len)
    pts_global = ego_xy[None, :] + unit[None, :] * ts[:, None]   # (route_len, 2)

    avails  = np.ones((1, route_len), dtype=np.bool_)
    pts_ego = vector_set_coordinates_to_local_frame(
        pts_global[np.newaxis], avails, anchor
    )[0]   # (route_len, 2)

    fwd = np.diff(pts_ego, axis=0)
    fwd = np.vstack([fwd, fwd[-1:]])

    tl   = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (route_len, 1))
    base = np.concatenate([
        pts_ego.astype(np.float32),
        fwd.astype(np.float32),
        np.zeros((route_len, 4), dtype=np.float32),
        tl,
    ], axis=1)   # (route_len, 12)

    route_lanes = np.zeros((route_num, route_len, 12), dtype=np.float32)
    for i in range(route_num):
        route_lanes[i] = base.copy()
    return route_lanes, pts_ego


def test_2_route_lanes_direction():
    """
    Scenario: CARLA spawn at (285.6, -243.7, yaw=90°).
    Destination 60 m ahead in CARLA = (285.6, -183.7).

    After standard conversion (pass-through), anchor heading = π/2.
    So 'forward' in ego frame = global +y direction.

    The first route-lane point [0] should be near (0,0) in ego frame (ego=origin).
    The last route-lane point [-1] should have a LARGE POSITIVE ego-x
    (because forward = +x in ego frame, and the route goes forward).
    """
    print("\n=== Test 2: route lane direction for CARLA spawn scenario ===")

    # CARLA spawn: x=285.6, y=-243.7, yaw=90°
    # Destination: straight ahead 60m → CARLA (285.6, -183.7)
    ego_cx, ego_cy, ego_yaw_deg = 285.6, -243.7, 90.0
    dst_cx, dst_cy              = 285.6, -183.7

    anchor  = np.array(carla_transform_to_standard(ego_cx, ego_cy, ego_yaw_deg))
    dest_std = np.array(carla_transform_to_standard(dst_cx, dst_cy, 0.0)[:2])

    print(f"  anchor std: {anchor}  (heading = {np.rad2deg(anchor[2]):.1f}°)")
    print(f"  dest std:   {dest_std}")
    print(f"  diff in std frame: {dest_std - anchor[:2]}")

    route_lanes, pts_ego = _make_straight_route_lanes_no_shift(
        anchor, dest_std
    )

    print(f"\n  pts_ego[0]  = {pts_ego[0]}   (should be ≈ (0, 0))")
    print(f"  pts_ego[5]  = {pts_ego[5]}")
    print(f"  pts_ego[-1] = {pts_ego[-1]}  (should be large +x, small |y|)")

    all_ok = True
    # First point: ego is at origin in ego frame
    all_ok &= check(np.linalg.norm(pts_ego[0]) < 2.0,
                    "pts_ego[0] is near origin",
                    f"got {pts_ego[0]}")

    # Last point: must be in front (positive ego-x) since destination is ahead
    all_ok &= check(pts_ego[-1, 0] > 10.0,
                    "pts_ego[-1] has large positive ego-x (route goes FORWARD)",
                    f"got ego-x={pts_ego[-1, 0]:.2f}")

    # Lateral drift: should be near 0 for straight-ahead destination
    all_ok &= check(abs(pts_ego[-1, 1]) < 3.0,
                    "pts_ego[-1] has small ego-y (no lateral drift for straight route)",
                    f"got ego-y={pts_ego[-1, 1]:.2f}")

    # fwd vectors should point in +x direction (forward)
    rl0 = route_lanes[0]  # (route_len, 12)
    fwd_x = rl0[:, 2]     # direction x component (cols 2-3 are fwd)
    fwd_y = rl0[:, 3]
    print(f"\n  fwd vector at pt 0: dx={fwd_x[0]:.3f}, dy={fwd_y[0]:.3f}  (expect dx>0, |dy|≈0)")
    print(f"  fwd vector at pt 5: dx={fwd_x[5]:.3f}, dy={fwd_y[5]:.3f}")

    all_ok &= check(fwd_x[0] > 0.1 and abs(fwd_y[0]) < 0.5,
                    "fwd direction at pt 0 is forward (+x ego, not sideways)",
                    f"got dx={fwd_x[0]:.3f} dy={fwd_y[0]:.3f}")
    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3 ─ the +10 shift bug check
# ═══════════════════════════════════════════════════════════════════════════════

def test_3_x_shift_effect():
    """
    The current code does pts_ego_shifted[:, 0] += 10.0 on the position columns.
    This shifts ALL route waypoints 10 m in the ego-x (forward) direction.

    This is fine IF the fwd vector is also computed from pts_ego_shifted.
    BUT the current code computes fwd from pts_ego (BEFORE the shift), 
    so positions and directions are inconsistent.

    More importantly: does the +10 shift change which *direction* the route appears
    to go? It shouldn't (it's a pure translation), but let's verify.

    We also check whether the shift value (10 m) matches the training data mean.
    If the model was trained with route points whose x-mean ≈ 10 m in ego frame,
    then shifting makes sense — but if the route is 60 m forward, the mean is
    already ~30 m in ego-x, not 10.
    """
    print("\n=== Test 3: effect of the +10 x-shift on route direction ===")

    ego_cx, ego_cy, ego_yaw_deg = 285.6, -243.7, 90.0
    dst_cx, dst_cy              = 285.6, -183.7

    anchor   = np.array(carla_transform_to_standard(ego_cx, ego_cy, ego_yaw_deg))
    dest_std = np.array(carla_transform_to_standard(dst_cx, dst_cy, 0.0)[:2])

    # -- without shift --
    _, pts_ego = _make_straight_route_lanes_no_shift(anchor, dest_std)
    mean_x_no_shift  = pts_ego[:, 0].mean()
    mean_y_no_shift  = pts_ego[:, 1].mean()

    # -- with shift --
    pts_ego_shifted = pts_ego.copy()
    pts_ego_shifted[:, 0] += 10.0
    mean_x_shifted  = pts_ego_shifted[:, 0].mean()

    print(f"  Without shift: mean ego-x={mean_x_no_shift:.2f}, mean ego-y={mean_y_no_shift:.2f}")
    print(f"  With +10 shift: mean ego-x={mean_x_shifted:.2f}")
    print(f"  Training distribution mean the shift targets: presumably ≈ 10 m")
    print(f"  Actual mean WITHOUT shift is {mean_x_no_shift:.1f} m "
          f"(route is {np.linalg.norm(dest_std - anchor[:2]):.0f}m long)")

    all_ok = True

    # Check: fwd direction BEFORE shift
    fwd_before = np.diff(pts_ego, axis=0)
    fwd_before = np.vstack([fwd_before, fwd_before[-1:]])

    # Check: fwd direction AFTER shift (same, because shift is constant)
    fwd_after = np.diff(pts_ego_shifted, axis=0)
    fwd_after  = np.vstack([fwd_after, fwd_after[-1:]])

    print(f"\n  fwd[0] before shift: dx={fwd_before[0,0]:.4f}, dy={fwd_before[0,1]:.4f}")
    print(f"  fwd[0] after  shift: dx={fwd_after[0,0]:.4f}, dy={fwd_after[0,1]:.4f}")
    all_ok &= check(np.allclose(fwd_before, fwd_after),
                    "+10 shift does NOT change fwd vectors (good)",
                    "")

    # THE BUG: in diff_adapter.py, fwd is computed from pts_ego (BEFORE shift)
    # but fed alongside pts_ego_SHIFTED. Since diff(shifted) == diff(original),
    # the fwd vectors are consistent whether computed before or after. This is NOT the bug.
    print("\n  [INFO] The +10 shift is consistent (fwd vectors unchanged).")
    print(f"  [INFO] However, the route mean ego-x={mean_x_no_shift:.1f}m >> 10m,")
    print(f"         so adding 10 pushes mean to {mean_x_shifted:.1f}m.")
    print(f"         If training distribution expects ego-x≈10m, shift may be WRONG direction/magnitude.")

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4 ─ spot-check coord_utils passthrough vs expected CARLA convention
# ═══════════════════════════════════════════════════════════════════════════════

def test_4_carla_yaw_convention():
    """
    CARLA yaw=0   → car faces +X in CARLA (East)
    CARLA yaw=90° → car faces +Y in CARLA (South in left-handed, or North in right-handed?)

    In CARLA's left-handed system:
        yaw=0   → forward = +X
        yaw=90° → forward = +Y  (which is LEFT in right-handed, but CARLA is left-handed)

    coord_utils.carla_transform_to_standard is currently a pass-through:
        std_x = carla_x, std_y = carla_y, std_h = deg2rad(carla_yaw)

    So we are treating CARLA's left-handed frame as right-handed.
    In nuPlan's right-handed frame:
        heading=0   → forward = +x
        heading=π/2 → forward = +y

    This test checks whether the spawn scenario (yaw=90°, dest forward in Y)
    correctly maps the route to ego-forward (+x) in ego frame.
    """
    print("\n=== Test 4: CARLA yaw=90° forward vector in ego frame ===")

    # CARLA: ego facing +Y (yaw=90°), dest 60m in +Y direction
    # Standard (pass-through): ego heading=π/2, so forward = +y in global standard
    # After ego transform: global +y → ego +x

    anchor_h90 = np.array([0.0, 0.0, np.pi / 2])  # ego at origin, heading=90°
    avails = np.ones((1, 1), dtype=np.bool_)

    def ego_local(global_xy):
        pt = np.array([[global_xy]], dtype=np.float64)
        return vector_set_coordinates_to_local_frame(pt, avails, anchor_h90)[0, 0]

    fwd_pt   = ego_local([0.0, 60.0])   # 60m ahead (global +y, since heading=90°)
    right_pt = ego_local([10.0, 0.0])   # 10m to global +x (right of heading=90° car)

    print(f"  Global +y (60m ahead for heading=90°) → ego: {fwd_pt}  [expect ~(+60, 0)]")
    print(f"  Global +x (10m to right for heading=90°) → ego: {right_pt}  [expect ~(0, -10)]")

    all_ok = True
    all_ok &= check(fwd_pt[0] > 50.0 and abs(fwd_pt[1]) < 2.0,
                    "CARLA yaw=90° forward (global +y) → ego +x",
                    f"got {fwd_pt}")
    all_ok &= check(right_pt[0] < 1.0 and right_pt[1] < -5.0,
                    "CARLA yaw=90° right (global +x) → ego -y",
                    f"got {right_pt}")

    if not all_ok:
        print("\n  *** HEADING CONVENTION MISMATCH ***")
        print("  If forward global +y does NOT map to ego +x with heading=π/2,")
        print("  then carla_transform_to_standard needs to FLIP the heading sign:")
        print("      std_h = -np.deg2rad(carla_yaw_deg)    # negate for left→right-hand")
        print("  OR the route generation needs to account for the flip.")

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5 ─ end-to-end: does the route_lanes tensor actually point forward?
# ═══════════════════════════════════════════════════════════════════════════════

def test_5_route_lane_endpoint():
    """
    Final sanity: with CARLA yaw=90° and destination 60m ahead in +Y,
    the LAST route-lane point (in ego frame) should be:
        ego-x ≈ +60–75m  (forward direction)
        ego-y ≈ 0        (no lateral drift)

    If instead we get:
        ego-x ≈ 0 and ego-y ≈ ±60m  → heading sign is WRONG (±90° off)
        ego-x ≈ small number          → route is a short path (perpendicular bug!)
    """
    print("\n=== Test 5: end-to-end route lane endpoint check ===")

    ego_cx, ego_cy, ego_yaw_deg = 285.6, -243.7, 90.0
    dst_cx, dst_cy              = 285.6, -183.7   # 60m ahead in CARLA +Y

    anchor   = np.array(carla_transform_to_standard(ego_cx, ego_cy, ego_yaw_deg))
    dest_std = np.array(carla_transform_to_standard(dst_cx, dst_cy, 0.0)[:2])

    print(f"  CARLA ego : ({ego_cx}, {ego_cy}, yaw={ego_yaw_deg}°)")
    print(f"  CARLA dest: ({dst_cx}, {dst_cy})")
    print(f"  std anchor: {anchor}")
    print(f"  std dest  : {dest_std}")

    dist_global = np.linalg.norm(dest_std - anchor[:2])
    print(f"  Distance in standard frame: {dist_global:.2f} m  (expect ~60 m)")

    route_lanes, pts_ego = _make_straight_route_lanes_no_shift(anchor, dest_std)

    last = pts_ego[-1]
    first = pts_ego[0]
    print(f"\n  Route lane pts_ego[0]  = ({first[0]:.2f}, {first[1]:.2f})")
    print(f"  Route lane pts_ego[-1] = ({last[0]:.2f}, {last[1]:.2f})")
    print(f"  (ego-x = forward, ego-y = left)")

    all_ok = True
    all_ok &= check(abs(dist_global - 60.0) < 5.0,
                    "distance in standard frame ≈ 60 m",
                    f"got {dist_global:.2f} m")
    all_ok &= check(last[0] > 50.0,
                    "last route pt ego-x > 50m (route points FORWARD)",
                    f"got ego-x={last[0]:.2f}")
    all_ok &= check(abs(last[1]) < 5.0,
                    "last route pt |ego-y| < 5m (no large lateral component)",
                    f"got ego-y={last[1]:.2f}")

    if not all_ok:
        # Diagnose what went wrong
        dx, dy = dest_std[0] - anchor[0], dest_std[1] - anchor[1]
        print(f"\n  DIAGNOSIS: diff in standard frame = ({dx:.2f}, {dy:.2f})")
        print(f"  anchor heading = {np.rad2deg(anchor[2]):.1f}°")
        if abs(last[0]) < 5.0 and abs(last[1]) > 20.0:
            print("  → Route is PERPENDICULAR to car (classic sign-flip bug!)")
            print("    FIX: negate heading in carla_transform_to_standard:")
            print("         std_h = -np.deg2rad(heading_deg)")
        elif last[0] < -10.0:
            print("  → Route points BEHIND the car (180° flip)")
            print("    FIX: add π to heading or negate y-coordinate")
        elif abs(last[0]) < 5.0 and abs(last[1]) < 5.0:
            print("  → Route is collapsed to near origin (destination too close?)")

    return all_ok



# ═══════════════════════════════════════════════════════════════════════════════
# Test 6 ─ simulate what diff_adapter actually sends to the model
# ═══════════════════════════════════════════════════════════════════════════════

def test_6_diff_adapter_route_simulation():
    """
    Replicate exactly what diff_adapter._make_straight_route_lanes does,
    INCLUDING the +10 x-shift, and check that the route still points forward.

    This is the TRUE test of the live code path.
    """
    print("\n=== Test 6: simulate diff_adapter._make_straight_route_lanes (with +10 shift) ===")

    ego_cx, ego_cy, ego_yaw_deg = 285.6, -243.7, 90.0
    dst_cx, dst_cy              = 285.6, -183.7

    anchor   = np.array(carla_transform_to_standard(ego_cx, ego_cy, ego_yaw_deg))
    dest_std = np.array(carla_transform_to_standard(dst_cx, dst_cy, 0.0)[:2])

    # ── exact replica of diff_adapter._make_straight_route_lanes ────────────
    route_num   = 25
    route_len   = 20
    overshoot_m = 15.0

    ego_xy = anchor[:2]
    diff   = dest_std - ego_xy
    dist   = np.linalg.norm(diff)
    unit   = diff / dist
    total  = dist + overshoot_m
    ts     = np.linspace(0.0, total, route_len)
    pts_global = ego_xy[None, :] + unit[None, :] * ts[:, None]

    avails  = np.ones((1, route_len), dtype=np.bool_)
    pts_ego = vector_set_coordinates_to_local_frame(
        pts_global[np.newaxis], avails, anchor
    )[0]

    pts_ego_shifted       = pts_ego.copy()
    pts_ego_shifted[:, 0] += 10.0   # THE +10 SHIFT

    fwd = np.diff(pts_ego, axis=0)          # computed from UNSHIFTED
    fwd = np.vstack([fwd, fwd[-1:]])

    tl   = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (route_len, 1))
    base = np.concatenate([
        pts_ego_shifted.astype(np.float32),  # POSITIONS use shifted
        fwd.astype(np.float32),              # FWD vectors from unshifted (same result anyway)
        np.zeros((route_len, 4), dtype=np.float32),
        tl,
    ], axis=1)

    print(f"  anchor: std({anchor[0]:.1f},{anchor[1]:.1f},{np.rad2deg(anchor[2]):.1f}°)")
    print(f"  dest:   std({dest_std[0]:.1f},{dest_std[1]:.1f})")
    print(f"  route pts (ego frame, BEFORE +10 shift):")
    print(f"    [0]: ({pts_ego[0,0]:.2f}, {pts_ego[0,1]:.2f})")
    print(f"    [-1]: ({pts_ego[-1,0]:.2f}, {pts_ego[-1,1]:.2f})")
    print(f"  route pts (ego frame, AFTER +10 shift on x):")
    print(f"    [0]: ({pts_ego_shifted[0,0]:.2f}, {pts_ego_shifted[0,1]:.2f})")
    print(f"    [-1]: ({pts_ego_shifted[-1,0]:.2f}, {pts_ego_shifted[-1,1]:.2f})")
    print(f"  fwd vector [0]: ({fwd[0,0]:.3f}, {fwd[0,1]:.3f})  [expect (+, ~0)]")

    all_ok = True
    all_ok &= check(pts_ego_shifted[-1, 0] > 50.0,
                    "With +10 shift: last pt ego-x > 50m (still points forward)",
                    f"got ego-x={pts_ego_shifted[-1, 0]:.2f}")
    all_ok &= check(abs(pts_ego_shifted[-1, 1]) < 5.0,
                    "With +10 shift: last pt |ego-y| < 5m (not sideways)",
                    f"got ego-y={pts_ego_shifted[-1, 1]:.2f}")
    all_ok &= check(fwd[0, 0] > 0.1 and abs(fwd[0, 1]) < 0.5,
                    "fwd direction is forward (+ego-x)",
                    f"got ({fwd[0,0]:.3f}, {fwd[0,1]:.3f})")

    # ── Now check: what does route_lane col 0 (position x) look like? ───────
    # Model was trained with route lane pts that have x≈? in ego frame.
    # The +10 shift was added to match training. Print mean for inspection.
    pos_x_mean = pts_ego_shifted[:, 0].mean()
    pos_y_mean = pts_ego_shifted[:, 1].mean()
    print(f"\n  AFTER shift: route-lane mean ego-x={pos_x_mean:.2f}, mean ego-y={pos_y_mean:.2f}")
    print(f"  If model expects mean ego-x≈10m but gets {pos_x_mean:.1f}m, route may be ignored.")
    if pos_x_mean > 30.0:
        print("  WARNING: mean ego-x >> 10m — the +10 shift may be wrong for long routes.")
        print("  Consider: remove the shift, or compute actual training distribution mean.")

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Route Lane Diagnostic Test")
    print("  (no model, no GPU, no CARLA required)")
    print("=" * 60)

    r1 = test_1_ego_frame_convention()
    r2 = test_2_route_lanes_direction()
    r3 = test_3_x_shift_effect()
    r4 = test_4_carla_yaw_convention()
    r5 = test_5_route_lane_endpoint()
    r6 = test_6_diff_adapter_route_simulation()

    print(f"\n{'=' * 60}")
    results = [r1, r2, r3, r4, r5, r6]
    passed  = sum(results)
    names   = ["ego-frame convention", "route direction", "+10 shift effect",
               "CARLA yaw convention", "end-to-end endpoint",
               "diff_adapter simulation (with +10 shift)"]
    for i, (name, ok) in enumerate(zip(names, results)):
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  {status} Test {i+1}: {name}")

    print(f"\n{passed}/{len(results)} tests passed")
    if passed == len(results):
        print("\nRoute lane geometry looks correct — issue is likely in model/guidance.")
    else:
        print("\nRoute lane geometry has problems — fix before running in CARLA.")
    print()
    print("NEXT STEP: if all pass here, run test_model_smoke.py with the checkpoint")
    print("  to check that model output (zeroed inputs) gives sensible trajectories.")
    print("  Then run test_coords.py to check direction for real destinations.")
