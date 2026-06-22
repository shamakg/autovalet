"""
test_guidance_suppression.py
----------------------------
Targeted diagnosis of the aisle-phase East-drift problem.

ROOT CAUSE CHAIN (what these tests verify):

  Step 1  goal_energy carries a strong East gradient at the final trajectory
          waypoint throughout the aisle phase.  path_energy at that same
          waypoint points BACKWARD (toward the L-corner), so it does NOT
          cancel the lateral pull.

  Step 2  Over multiple DPM steps and ticks, the uncancelled East gradient
          rotates the car's heading from π/2 (South) toward 0 (East).

  Step 3  The anchor-rotation trigger fires when dest_ego_x < TURN_X_THRESHOLD.
          But dest_ego_x is computed in the car's CURRENT heading frame.
          With a drifted heading (e.g. 0.4 rad), the destination appears
          "still ahead" (dest_ego_x ≈ 7 m) even at the junction row, so
          the trigger never fires — the car just drives East through the
          wrong row.

FIX: suppress guidance while dest_ego_x >= TURN_X_THRESHOLD (computed with
     the ACTUAL, non-rotated heading).  Without guidance, route conditioning
     drives the car straight South; heading stays ≈ π/2 at the junction, so
     the trigger fires correctly and anchor rotation works.

No model, no CARLA server, no GPU required.
"""

import sys, os
import numpy as np
import torch

_HERE         = os.path.dirname(os.path.abspath(__file__))
_ADAPTER_ROOT = os.path.dirname(_HERE)
_AUTOVALET    = os.path.dirname(_ADAPTER_ROOT)
_NUPLAN_ROOT  = os.path.join(_ADAPTER_ROOT, "nuplan-devkit")
_DP_ROOT      = os.path.join(_ADAPTER_ROOT, "Diffusion-Planner")

_CARLA_ROOT    = "/home/sumesh/opt/carla/PythonAPI/carla"
_SCENARIO_ROOT = "/home/sumesh/carla_garage/scenario_runner"
_LB_ROOT       = "/home/sumesh/carla_garage/leaderboard"

for _p in [_CARLA_ROOT, _SCENARIO_ROOT, _LB_ROOT,
           _HERE, _AUTOVALET, _ADAPTER_ROOT, _NUPLAN_ROOT, _DP_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import diff_adapter as da
from diffusion_planner.utils.config import Config
from diffusion_planner.data_process.utils import _state_se2_array_to_transform_matrix
from utils.coord_utils import carla_transform_to_standard

CKPT = os.path.join(_ADAPTER_ROOT, "checkpoints/model.pth")
ARGS = os.path.join(_ADAPTER_ROOT, "checkpoints/args.json")

# ── Spot-17 scenario (Town04) ────────────────────────────────────────────────
# carla_transform_to_standard is a passthrough: std = CARLA
EGO_SPAWN_CX,  EGO_SPAWN_CY,  EGO_YAW = 285.6, -243.73, 90.0   # South
DEST_CX,       DEST_CY               = 293.3,  -232.73
DEST_ANGLE_DEG                        = 0.0    # spot faces East in CARLA

# nuPlan normaliser constants (match diff_adapter _guidance_fn)
MEAN_XY = np.array([10.0, 0.0])
STD_XY  = np.array([20.0, 20.0])

# These match the constants in diff_adapter.py
GOAL_SCALE       = da.GOAL_SCALE
PATH_SCALE       = da.PATH_SCALE
TURN_X_THRESHOLD = da.TURN_X_THRESHOLD

# ── Helpers ──────────────────────────────────────────────────────────────────

def ok(tag):         print(f"  [PASS] {tag}")
def fail(tag, msg): print(f"  [FAIL] {tag}  →  {msg}"); return False
def check(cond, tag, msg=""):
    if cond: ok(tag); return True
    else:    return fail(tag, msg)


def _dest_in_ego(ego_std_pos, ego_heading_rad, dest_std):
    """Return (dest_ego_x, dest_ego_y) for given ego pose."""
    dx = dest_std[0] - ego_std_pos[0]
    dy = dest_std[1] - ego_std_pos[1]
    h  = ego_heading_rad
    ego_x =  np.cos(h) * dx + np.sin(h) * dy
    ego_y = -np.sin(h) * dx + np.cos(h) * dy
    return ego_x, ego_y


def _setup_globals():
    """Prime diff_adapter globals (no model, no CARLA). Returns (anchor, config)."""
    config = Config(ARGS, guidance_fn=None)
    da._config = config

    ax, ay, ah = carla_transform_to_standard(EGO_SPAWN_CX, EGO_SPAWN_CY, EGO_YAW)
    dx, dy, _  = carla_transform_to_standard(DEST_CX, DEST_CY, 0.0)

    da._start_std      = np.array([ax, ay])
    da._dest_std       = np.array([dx, dy])
    da._dest_angle_std = float(np.deg2rad(DEST_ANGLE_DEG))
    da._route_pts_ego  = None

    return np.array([ax, ay, ah]), config


# ── Pure-energy helpers (no model: skip Tweedie) ─────────────────────────────

def _goal_energy_grad_at_final(final_real_np, dest_ego_np):
    """
    Compute goal_energy and its gradient w.r.t. the normalised final waypoint.

    The DPM solver passes *normalised* x to the guidance function; internally
    _guidance_fn denormalises to real metres before computing energy.
    We replicate that: convert real→norm for the grad-tracked tensor, then
    denorm inside the energy formula.  Gradient returned is ∂energy/∂x_norm.

    Parameters match _guidance_fn formula exactly:
        energy = -(GOAL_SCALE / STD_XY[0]) * ||final_real - dest_ego||
    """
    mxy = torch.tensor(MEAN_XY, dtype=torch.float32)
    sxy = torch.tensor(STD_XY,  dtype=torch.float32)
    dst = torch.tensor(dest_ego_np, dtype=torch.float32)

    # Leaf tensor in normalised space (as x appears in the DPM solver)
    final_norm_np = (final_real_np - MEAN_XY) / STD_XY
    fn = torch.tensor(final_norm_np, dtype=torch.float32, requires_grad=True)

    fn_real     = fn * sxy + mxy          # round-trips back to final_real_np
    goal_dist   = torch.norm(fn_real - dst)
    goal_energy = -(GOAL_SCALE / STD_XY[0]) * goal_dist
    goal_energy.backward()
    return goal_energy.item(), fn.grad.detach().numpy().copy()


def _path_energy_grad_at_point(point_real_np, route_pts_ego):
    """
    Compute path_energy and its gradient w.r.t. a single normalised waypoint.

    energy = -(PATH_SCALE / STD_XY[0]) * min_dist_to_route
    """
    pn  = torch.tensor((point_real_np - MEAN_XY) / STD_XY, dtype=torch.float32, requires_grad=True)
    rt  = torch.tensor(route_pts_ego,  dtype=torch.float32)
    mxy = torch.tensor(MEAN_XY,        dtype=torch.float32)
    sxy = torch.tensor(STD_XY,         dtype=torch.float32)

    pt_real   = pn * sxy + mxy
    diff       = pt_real[None, :] - rt            # (M, 2)
    dist       = torch.norm(diff, dim=-1)         # (M,)
    min_dist   = dist.min()
    path_energy = -(PATH_SCALE / STD_XY[0]) * min_dist
    path_energy.backward()
    return path_energy.item(), pn.grad.detach().numpy().copy()


# ── Test 1: goal_energy has a large East gradient at the final waypoint ──────

def test_goal_energy_east_gradient_in_aisle():
    """
    At spawn (South-facing), dest_ego = (11m, -7.7m).
    A "straight-ahead" trajectory ends at ≈ (15m, 0) in ego-real space.
    goal_energy's gradient at that final waypoint must have:
      - Large |y-component| pointing East (negative ego-y)
      - |grad_y| > |grad_x|  (lateral pull dominates forward pull)
    This is the energy term driving premature East steering.
    """
    print("\n=== Test 1: goal_energy gradient is East-dominant at final waypoint (aisle) ===")

    ax, ay, ah = carla_transform_to_standard(EGO_SPAWN_CX, EGO_SPAWN_CY, EGO_YAW)
    dx, dy, _  = carla_transform_to_standard(DEST_CX,      DEST_CY,       0.0)

    dest_ego_x, dest_ego_y = _dest_in_ego(
        np.array([ax, ay]), ah, np.array([dx, dy])
    )
    print(f"  dest_ego (spawn, South-facing): ({dest_ego_x:.2f}, {dest_ego_y:.2f}) m")

    # Model predicts ≈ 15m ahead at t=1s — representative "straight-ahead" final waypoint
    final_real = np.array([15.0, 0.0])
    dest_ego   = np.array([dest_ego_x, dest_ego_y])

    energy, grad = _goal_energy_grad_at_final(final_real, dest_ego)
    print(f"  goal_energy          : {energy:.1f}")
    print(f"  grad w.r.t. norm[x,y]: [{grad[0]:.1f}, {grad[1]:.1f}]")
    print(f"  |grad_y| / |grad_x|  : {abs(grad[1]) / max(abs(grad[0]), 1e-6):.2f}x")

    all_ok = True
    all_ok &= check(grad[1] < -50,
                    f"grad_y = {grad[1]:.1f} < -50  (East pull is large and negative)",
                    "goal_energy gradient does NOT point East in aisle — hypothesis wrong")
    all_ok &= check(abs(grad[1]) > abs(grad[0]),
                    f"|grad_y| ({abs(grad[1]):.0f}) > |grad_x| ({abs(grad[0]):.0f})  "
                    "(lateral component dominates)",
                    "forward pull dominates — East drift would be smaller than expected")
    return all_ok


# ── Test 2: path_energy at the final waypoint is BACKWARD, not lateral ───────

def test_path_energy_no_lateral_cancel():
    """
    At spawn (South-facing), the L-shaped route goes:
        aisle: (0,0) → (11,0)   [ego frame, straight South]
        turn:  (11,0) → (11,-7.7)

    For a trajectory final waypoint at (15m, 0), the nearest route point is
    the L-corner (11, 0) — 4m behind, no lateral offset.

    Therefore path_energy's gradient at the final waypoint is:
      - Purely BACKWARD (negative ego-x), no East component
      - It does NOT cancel goal_energy's East pull

    This means goal_energy's East gradient is completely unopposed at the final
    waypoint, explaining why the car drifts East.
    """
    print("\n=== Test 2: path_energy gradient at final waypoint is backward, not East ===")

    anchor, _ = _setup_globals()
    da._make_straight_route_lanes(anchor)
    route = da._route_pts_ego
    assert route is not None

    final_real = np.array([15.0, 0.0])  # straight-ahead, 1m past the L-corner

    # Find the nearest route point manually (for diagnostic print)
    dists         = np.linalg.norm(route - final_real[None, :], axis=1)
    nearest_idx   = int(np.argmin(dists))
    nearest_pt    = route[nearest_idx]
    nearest_dist  = float(dists[nearest_idx])
    print(f"  nearest route pt to (15,0): ({nearest_pt[0]:.2f}, {nearest_pt[1]:.2f})  "
          f"dist={nearest_dist:.2f}m")

    energy, grad = _path_energy_grad_at_point(final_real, route)
    print(f"  path_energy          : {energy:.1f}")
    print(f"  grad w.r.t. norm[x,y]: [{grad[0]:.1f}, {grad[1]:.1f}]")

    # Also compute goal_energy gradient to show the East pull is unopposed
    ax, ay, ah = carla_transform_to_standard(EGO_SPAWN_CX, EGO_SPAWN_CY, EGO_YAW)
    dx, dy, _  = carla_transform_to_standard(DEST_CX, DEST_CY, 0.0)
    dest_ego_x, dest_ego_y = _dest_in_ego(np.array([ax, ay]), ah, np.array([dx, dy]))
    _, g_grad = _goal_energy_grad_at_final(final_real, np.array([dest_ego_x, dest_ego_y]))

    net_grad_y = g_grad[1] + grad[1]
    print(f"  goal_grad_y = {g_grad[1]:.0f}  (East pull)")
    print(f"  path_grad_y = {grad[1]:.0f}  (near-zero lateral)")
    print(f"  net_grad_y  = {net_grad_y:.0f}  (still strongly East)")

    all_ok = True
    all_ok &= check(abs(grad[1]) < 50,
                    f"grad_y = {grad[1]:.1f}: path_energy has NO lateral pull at final waypoint",
                    "path_energy has lateral gradient at final waypoint — it might cancel East pull")
    all_ok &= check(abs(grad[0]) > abs(grad[1]),
                    f"|grad_x| ({abs(grad[0]):.0f}) > |grad_y| ({abs(grad[1]):.0f}): "
                    "gradient is backward, not East",
                    "path_energy is not backward — geometry different than expected")
    all_ok &= check(abs(net_grad_y) > abs(g_grad[1]) * 0.8,
                    f"Net East pull ({abs(net_grad_y):.0f}) is ≥80% of goal_grad_y alone "
                    "— path_energy does not cancel it",
                    "path_energy actually cancels most of the East pull — re-examine hypothesis")
    return all_ok


# ── Test 3: heading drift prevents the anchor-rotation trigger ────────────────

def test_trigger_fails_with_heading_drift():
    """
    The anchor-rotation trigger fires when dest_ego_x < TURN_X_THRESHOLD.
    dest_ego_x depends on the car's CURRENT heading.

    At the junction row (y=-232.73 CARLA):
      South-facing (h=π/2): dest_ego_x = 0.0  →  trigger fires   ✓
      East-drifted (h=0.4): dest_ego_x = 7.09 →  trigger silent   ✗

    This is why the car never turns into the spot after guidance-induced drift.
    """
    print("\n=== Test 3: heading drift prevents anchor-rotation trigger at junction ===")

    dx, dy, _ = carla_transform_to_standard(DEST_CX, DEST_CY, 0.0)
    dest_std  = np.array([dx, dy])

    jx, jy = EGO_SPAWN_CX, -232.73   # junction row, same column as spawn

    print(f"  Junction CARLA: ({jx}, {jy})")
    print(f"  Dest    CARLA:  ({DEST_CX}, {DEST_CY})")
    print(f"  TURN_X_THRESHOLD: {TURN_X_THRESHOLD} m")
    print()

    results = {}
    for label, heading_rad in [("South  (h=π/2, correct)", np.pi/2),
                                ("Drifted (h=0.4 rad, 67° toward East)", 0.4)]:
        jx_std, jy_std, _ = carla_transform_to_standard(jx, jy, np.rad2deg(heading_rad))
        ego_x, ego_y = _dest_in_ego(np.array([jx_std, jy_std]), heading_rad, dest_std)
        fires = ego_x < TURN_X_THRESHOLD
        print(f"  heading={label}")
        print(f"    dest_ego_x = {ego_x:.2f} m   trigger={'FIRES ✓' if fires else 'SILENT ✗'}")
        results[label] = (ego_x, fires)

    south_x, south_fires = results["South  (h=π/2, correct)"]
    drift_x, drift_fires = results["Drifted (h=0.4 rad, 67° toward East)"]

    all_ok = True
    all_ok &= check(south_fires,
                    f"South heading: dest_ego_x={south_x:.2f} < {TURN_X_THRESHOLD} → trigger fires",
                    "Trigger should fire when car is correctly South-facing at junction")
    all_ok &= check(not drift_fires,
                    f"Drifted heading: dest_ego_x={drift_x:.2f} >= {TURN_X_THRESHOLD} → trigger silent",
                    "Trigger should NOT fire with drifted heading (proves the break)")
    all_ok &= check(drift_x > 5.0,
                    f"Drifted dest_ego_x={drift_x:.2f} >> threshold (trigger very far from firing)",
                    "Drift is not large enough to matter")
    return all_ok


# ── Test 4: suppression-flag logic (aisle→True, junction→False) ──────────────

def test_suppression_flag_logic():
    """
    The proposed fix sets:
        _suppress_guidance = (dest_ego_x >= TURN_X_THRESHOLD)
    computed with the ACTUAL (non-rotated) heading.

    For a South-facing car moving from spawn to junction:
      y=-243.73  dest_ego_x=11.0  → suppress=True   (in aisle)
      y=-240.00  dest_ego_x= 7.27 → suppress=True   (in aisle)
      y=-235.00  dest_ego_x= 2.27 → suppress=True   (in aisle)
      y=-233.73  dest_ego_x= 1.00 → suppress=True   (at threshold)
      y=-232.73  dest_ego_x= 0.00 → suppress=False  (at junction → activate)
      y=-231.73  dest_ego_x=-1.00 → suppress=False  (past junction → still active)
    """
    print("\n=== Test 4: suppression flag — True in aisle, False at and past junction ===")

    dx, dy, _ = carla_transform_to_standard(DEST_CX, DEST_CY, 0.0)
    dest_std  = np.array([dx, dy])
    h         = np.deg2rad(EGO_YAW)   # π/2 = South

    test_positions = [
        ("spawn",         EGO_SPAWN_CY,  True),
        ("mid-aisle",     -240.00,        True),
        ("near-junction", -235.00,        True),
        ("threshold",     -233.73,        True),   # dest_ego_x = 1.0 exactly
        ("junction",      -232.73,        False),
        ("past-junction", -231.73,        False),
    ]

    all_ok = True
    for label, car_cy, expected_suppress in test_positions:
        car_cx = EGO_SPAWN_CX
        cx_s, cy_s, _ = carla_transform_to_standard(car_cx, car_cy, np.rad2deg(h))
        ego_x, _      = _dest_in_ego(np.array([cx_s, cy_s]), h, dest_std)
        suppress      = (ego_x >= TURN_X_THRESHOLD)
        match         = suppress == expected_suppress
        status        = "✓" if match else "✗"
        print(f"  {label:15s}  car_y={car_cy:8.2f}  dest_ego_x={ego_x:6.2f}  "
              f"suppress={str(suppress):5s}  expected={str(expected_suppress):5s}  {status}")
        all_ok &= check(match,
                        f"{label}: suppress={suppress} (expected {expected_suppress})",
                        f"Flag logic wrong at y={car_cy}")

    return all_ok


# ── Test 5: at junction with East anchor, goal gradient is forward-only ───────

def test_guidance_at_junction_is_forward():
    """
    After anchor rotation (East heading at junction):
      - anchor position = junction row, heading = 0 (East)
      - dest_ego = (7.7, 0)  — destination is straight ahead in East frame

    goal_energy gradient at the final waypoint (e.g. (5m, 0) in East-frame ego):
      - Should point in +x direction (forward = East in world)
      - y-component should be ≈ 0 (no spurious lateral pull)

    This proves guidance is safe to activate at the junction after anchor rotation.
    """
    print("\n=== Test 5: guidance at junction (East anchor) is forward-only ===")

    # Junction row, East-heading anchor
    jx_std, jy_std, _ = carla_transform_to_standard(EGO_SPAWN_CX, -232.73, 0.0)  # East heading
    east_h            = 0.0                                   # East heading in std
    dx, dy, _         = carla_transform_to_standard(DEST_CX, DEST_CY, 0.0)

    dest_ego_x, dest_ego_y = _dest_in_ego(
        np.array([jx_std, jy_std]), east_h, np.array([dx, dy])
    )
    print(f"  East-anchor at junction: dest_ego = ({dest_ego_x:.2f}, {dest_ego_y:.2f}) m")

    # A trajectory ending at (5m, 0) in East-frame ego real space
    final_real = np.array([5.0, 0.0])
    dest_ego   = np.array([dest_ego_x, dest_ego_y])

    energy, grad = _goal_energy_grad_at_final(final_real, dest_ego)
    print(f"  goal_energy          : {energy:.1f}")
    print(f"  grad w.r.t. norm[x,y]: [{grad[0]:.1f}, {grad[1]:.1f}]")

    # Sign convention: gradient points toward higher energy = toward destination.
    # final=(5,0), dest=(7.7,0): destination is AHEAD, so grad_x is POSITIVE (push forward).
    # The DPM solver subtracts grad from noise, causing x_0 to move in grad direction.
    all_ok = True
    all_ok &= check(grad[0] > 10,
                    f"grad_x = {grad[0]:.1f} > 10  (positive forward pull toward destination)",
                    "No forward gradient at junction — guidance won't steer car into spot")
    all_ok &= check(abs(grad[1]) < 10,
                    f"|grad_y| = {abs(grad[1]):.2f} < 10  (minimal lateral gradient at junction)",
                    "Large lateral gradient at junction — guidance will cause lateral overshoot")
    all_ok &= check(abs(grad[0]) > abs(grad[1]) * 5,
                    f"|grad_x| ({abs(grad[0]):.1f}) >> |grad_y| ({abs(grad[1]):.2f})  "
                    "(forward strongly dominates)",
                    "Lateral component is not negligible at junction")
    return all_ok


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  Guidance suppression — root-cause diagnosis tests")
    print("=" * 65)

    r1 = test_goal_energy_east_gradient_in_aisle()
    r2 = test_path_energy_no_lateral_cancel()
    r3 = test_trigger_fails_with_heading_drift()
    r4 = test_suppression_flag_logic()
    r5 = test_guidance_at_junction_is_forward()

    print(f"\n{'='*65}")
    results = [r1, r2, r3, r4, r5]
    names   = [
        "Goal energy: East-dominant gradient in aisle",
        "Path energy: backward at final waypoint (no East cancel)",
        "Heading drift silences anchor-rotation trigger",
        "Suppression flag: True in aisle, False at junction",
        "Guidance at junction: forward-only, no lateral pull",
    ]
    passed = sum(bool(r) for r in results)
    for name, ok in zip(names, results):
        print(f"  {'[PASS]' if ok else '[FAIL]'} {name}")
    print(f"\n{passed}/{len(results)} tests passed")
    sys.exit(0 if passed == len(results) else 1)
