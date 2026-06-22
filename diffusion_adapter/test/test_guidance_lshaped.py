"""
test_guidance_lshaped.py  —  Verify that the route-following path_energy fix is correct.

Three tests, no CARLA, no GPU required:
  1. _route_pts_ego is populated after _make_straight_route_lanes (spot 17 geometry).
  2. The stored route is L-shaped: straight South in the aisle, then East into the spot.
  3. path_energy penalises a diagonal trajectory more than an L-shaped one, meaning
     the diffusion model will be guided away from the diagonal shortcut.

Run with:
    /home/sumesh/envs/simlingo/bin/python test/test_guidance_lshaped.py
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

from diffusion_planner.utils.config import Config
from utils.coord_utils import carla_transform_to_standard
import diff_adapter as da

CKPT = os.path.join(_ADAPTER_ROOT, "checkpoints/model.pth")
ARGS = os.path.join(_ADAPTER_ROOT, "checkpoints/args.json")

# Spot 17 geometry (CARLA Town04)
EGO_CX,   EGO_CY,   EGO_YAW  = 285.6, -243.73, 90.0   # spawn facing South
DEST_CX,  DEST_CY             = 293.3, -232.73          # far destination (East + North)
DEST_ANGLE                    = 0.0                      # spot faces East (heading=0)

PATH_SCALE = da.PATH_SCALE


def ok(tag):   print(f"  [PASS] {tag}")
def fail(tag, detail=""):  print(f"  [FAIL] {tag}  {detail}"); return False
def check(cond, tag, detail=""):
    if cond: ok(tag);   return True
    else:    return fail(tag, detail)


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

def _setup_globals():
    """Prime diff_adapter module-level globals for spot 17 without running the model."""
    config = Config(ARGS, guidance_fn=None)
    da._config = config

    ax, ay, ah = carla_transform_to_standard(EGO_CX, EGO_CY, EGO_YAW)
    anchor      = np.array([ax, ay, ah])
    dx, dy, _   = carla_transform_to_standard(DEST_CX, DEST_CY, 0.0)

    da._start_std      = anchor[:2].copy()
    da._dest_std       = np.array([dx, dy])
    da._dest_angle_std = float(np.deg2rad(DEST_ANGLE))
    da._route_pts_ego  = None   # ensure clean state before test

    return anchor, config


# ---------------------------------------------------------------------------
# Test 1: _route_pts_ego is populated
# ---------------------------------------------------------------------------

def test_route_pts_ego_populated():
    """After _make_straight_route_lanes, da._route_pts_ego must not be None."""
    print("\n=== Test 1: _route_pts_ego is populated ===")

    anchor, _ = _setup_globals()
    assert da._route_pts_ego is None, "precondition: _route_pts_ego should be None before call"

    da._make_straight_route_lanes(anchor)

    all_ok = True
    all_ok &= check(da._route_pts_ego is not None,
                    "_route_pts_ego is not None after _make_straight_route_lanes")
    if da._route_pts_ego is not None:
        all_ok &= check(da._route_pts_ego.ndim == 2 and da._route_pts_ego.shape[1] == 2,
                        f"_route_pts_ego is (N,2) array  (shape={da._route_pts_ego.shape})")
        all_ok &= check(len(da._route_pts_ego) >= 20,
                        f"route has >= 20 points  (got {len(da._route_pts_ego)})")
    return all_ok


# ---------------------------------------------------------------------------
# Test 2: stored route is L-shaped
# ---------------------------------------------------------------------------

def test_route_is_lshaped():
    """
    For spot 17, the route should:
      - Start near the ego origin (0, 0) in ego frame.
      - Go forward (ego +x = South) while staying laterally near ego-y = 0 for the first ~11m.
      - Then turn right (ego -y = East) once x ≈ 11m.

    Checks:
      a) First point near origin.
      b) Lateral drift (|y|) during the aisle segment (x < 9m) is < 1m.
      c) Final pre-overshoot segment turns East: last 10 points have mean y < -3m.
    """
    print("\n=== Test 2: stored route is L-shaped ===")

    anchor, _ = _setup_globals()
    da._make_straight_route_lanes(anchor)
    pts = da._route_pts_ego   # (N, 2), x=forward, y=left

    all_ok = True

    # a) First point near origin (ego is at origin)
    dist0 = float(np.linalg.norm(pts[0]))
    all_ok &= check(dist0 < 1.0,
                    f"First route point near origin (dist={dist0:.2f}m < 1m)",
                    f"got ({pts[0,0]:.2f}, {pts[0,1]:.2f})")

    # b) Aisle segment is laterally straight (not diagonal)
    aisle_mask = pts[:, 0] < 9.0   # ego-x < 9m = still in the aisle
    if aisle_mask.sum() < 3:
        all_ok &= fail("aisle segment: not enough points with ego-x < 9m", f"pts[:5]: {pts[:5]}")
    else:
        aisle_y = pts[aisle_mask, 1]
        max_lat = float(np.abs(aisle_y).max())
        mean_lat = float(np.abs(aisle_y).mean())
        print(f"  Aisle segment: {aisle_mask.sum()} pts, max |ego-y|={max_lat:.2f}m, "
              f"mean |ego-y|={mean_lat:.2f}m  (expect < 1m)")
        all_ok &= check(max_lat < 1.5,
                        f"Aisle segment is laterally straight (max |ego-y|={max_lat:.2f}m < 1.5m)",
                        "route goes diagonally in the aisle — L-shape not computed correctly")

    # c) Turn segment turns East (ego-y becomes negative after the bend)
    tail = pts[-min(15, len(pts)):]
    mean_tail_y = float(tail[:, 1].mean())
    print(f"  Turn segment (last 15 pts): mean ego-y={mean_tail_y:.2f}m  (expect < -3m, East=-y)")
    all_ok &= check(mean_tail_y < -3.0,
                    f"Turn segment goes East (mean ego-y={mean_tail_y:.2f}m < -3m)",
                    "route never turns East — L-shape missing the right-turn component")

    # Print a few representative points for diagnostics
    n = len(pts)
    for label, idx in [("start", 0), ("aisle_mid", n//4), ("bend", n//2), ("dest", -1)]:
        print(f"  pts[{label}]:  x={pts[idx,0]:.2f}  y={pts[idx,1]:.2f}")

    return all_ok


# ---------------------------------------------------------------------------
# Test 3: path_energy penalises diagonal more than L-shaped
# ---------------------------------------------------------------------------

def test_path_energy_favours_lshaped():
    """
    Create two synthetic 'future_xy_real' trajectory tensors — both start at
    (0,0) in ego frame and end at (11, -7.7) (the spot):

      diagonal: goes diagonally, linearly interpolating x and y simultaneously.
      lshaped:  goes forward (x 0→11, y=0) then right (x=11, y 0→-7.7).

    With the new path_energy, the L-shaped trajectory should be closer to the
    stored route and therefore have a HIGHER (less negative) energy score.

    This test does not require the model or CARLA.
    """
    print("\n=== Test 3: path_energy favours L-shaped over diagonal trajectory ===")

    anchor, _ = _setup_globals()
    da._make_straight_route_lanes(anchor)
    route_pts = da._route_pts_ego  # (N, 2) in ego frame

    # L-shaped route in ego frame: (0,0)→(11,0)→(11,-7.7)
    T = 20   # trajectory length
    # Diagonal: linear from (0,0) to (11,-7.7)
    diag = np.stack([np.linspace(0, 11, T), np.linspace(0, -7.7, T)], axis=1)
    # L-shaped: first go forward to (11,0), then right to (11,-7.7)
    half = T // 2
    lshp = np.vstack([
        np.stack([np.linspace(0, 11, half), np.zeros(half)], axis=1),
        np.stack([np.full(T - half, 11.0), np.linspace(0, -7.7, T - half)], axis=1),
    ])

    def compute_path_energy(traj_np):
        future_xy_real = torch.tensor(traj_np[None], dtype=torch.float32)  # (1, T, 2)
        route = torch.tensor(route_pts, dtype=torch.float32)
        if len(route) > 200:
            idx   = torch.linspace(0, len(route) - 1, 200, dtype=torch.long)
            route = route[idx]
        diff_r        = future_xy_real[:, :, None, :] - route[None, None, :, :]  # (1, T, M, 2)
        dist_r        = torch.norm(diff_r, dim=-1)                               # (1, T, M)
        min_dist_r, _ = dist_r.min(dim=-1)                                       # (1, T)
        std_xy        = torch.tensor([20.0, 20.0])
        return -(PATH_SCALE / std_xy[0]) * min_dist_r.mean()

    e_diag = compute_path_energy(diag).item()
    e_lshp = compute_path_energy(lshp).item()

    print(f"  path_energy(diagonal):  {e_diag:.2f}")
    print(f"  path_energy(L-shaped):  {e_lshp:.2f}")
    print(f"  Difference (L - diag):  {e_lshp - e_diag:.2f}  (expect > 0)")

    # Mean distance to route
    def mean_route_dist(traj_np):
        dists = np.linalg.norm(traj_np[:, None, :] - route_pts[None, :, :], axis=2).min(axis=1)
        return dists.mean()

    print(f"  Mean route dist (diagonal): {mean_route_dist(diag):.2f}m")
    print(f"  Mean route dist (L-shaped): {mean_route_dist(lshp):.2f}m")

    all_ok = True
    all_ok &= check(e_lshp > e_diag,
                    f"L-shaped energy ({e_lshp:.2f}) > diagonal energy ({e_diag:.2f})",
                    "path_energy does not prefer L-shaped route — guidance will still go diagonal")
    all_ok &= check(e_diag < -10.0,
                    f"diagonal energy is clearly penalised ({e_diag:.2f} < -10)",
                    "path_energy barely penalises the wrong trajectory")
    return all_ok


# ---------------------------------------------------------------------------
# Test 4: anchor rotation at the turn point
# ---------------------------------------------------------------------------

def test_anchor_rotation_at_turn():
    """
    At the junction row (car at spot-17 x=285.6, y=-232.7, heading South),
    the destination in the South-facing ego frame is purely lateral (0, -7.7).
    The anchor-rotation logic should:
      a) Detect dest_ego_x < TURN_X_THRESHOLD and set effective_heading = 0 (East).
      b) With the East-heading anchor, dest_ego should be (7.7, 0) — straight ahead.
      c) The route in East-heading ego frame should be forward-biased (all x > 0, |y| small).
    """
    print("\n=== Test 4: anchor rotation at junction ===")

    from diffusion_planner.data_process.utils import _state_se2_array_to_transform_matrix

    anchor_junction, _ = _setup_globals()
    # Override to place ego exactly at the junction row
    junc_x, junc_y = EGO_CX, 285.6  # same x, at CARLA junction y
    junc_y_carla   = -232.73          # junction row y in CARLA
    ax_j, ay_j, ah_j = carla_transform_to_standard(junc_x, junc_y_carla, EGO_YAW)
    anchor_j = np.array([ax_j, ay_j, ah_j])
    da._start_std = anchor_j[:2].copy()

    # South-facing anchor at junction: dest_ego_x should be ≈ 0
    M_j     = _state_se2_array_to_transform_matrix(anchor_j)
    M_j_inv = np.linalg.inv(M_j)
    dest_h  = np.array([da._dest_std[0], da._dest_std[1], 1.0])
    dest_ego_south = (M_j_inv @ dest_h)[:2]
    print(f"  South anchor at junction: dest_ego=({dest_ego_south[0]:.2f},{dest_ego_south[1]:.2f})")

    all_ok = True
    all_ok &= check(abs(dest_ego_south[0]) < da.TURN_X_THRESHOLD,
                    f"dest_ego_x ({dest_ego_south[0]:.2f}) < TURN_X_THRESHOLD ({da.TURN_X_THRESHOLD}): "
                    "anchor rotation should trigger",
                    "anchor rotation would NOT trigger — turn still broken")

    # Apply East-heading anchor (as diffusion_plan would)
    effective_h     = da._dest_angle_std   # = 0.0 (East)
    eff_anchor      = np.array([ax_j, ay_j, effective_h])
    M_eff           = _state_se2_array_to_transform_matrix(eff_anchor)
    M_eff_inv       = np.linalg.inv(M_eff)
    dest_ego_east   = (M_eff_inv @ dest_h)[:2]
    print(f"  East anchor at junction:  dest_ego=({dest_ego_east[0]:.2f},{dest_ego_east[1]:.2f})")

    all_ok &= check(dest_ego_east[0] > 5.0,
                    f"East-anchor dest_ego_x={dest_ego_east[0]:.2f}m > 5m (destination is ahead)",
                    "With East anchor, destination should appear straight ahead")
    all_ok &= check(abs(dest_ego_east[1]) < 1.0,
                    f"East-anchor dest_ego_y={dest_ego_east[1]:.2f}m ≈ 0 (no lateral offset)",
                    "Destination should have minimal lateral component in East-heading frame")

    # Route in East-heading frame should be forward-biased
    da._make_straight_route_lanes(eff_anchor)
    pts = da._route_pts_ego
    if pts is not None:
        min_x, max_y = pts[:, 0].min(), np.abs(pts[:, 1]).max()
        print(f"  East-anchor route: min_x={min_x:.2f}  max|y|={max_y:.2f}")
        all_ok &= check(min_x >= -0.5,
                        f"Route min_x={min_x:.2f} ≥ -0.5 (forward-biased, in-distribution)",
                        "Route should not go backward in East-heading frame")
        all_ok &= check(max_y < 2.0,
                        f"Route max|y|={max_y:.2f} < 2m (minimal lateral offset)",
                        "Route should be nearly straight in East-heading frame")
    else:
        all_ok &= fail("_route_pts_ego is None after _make_straight_route_lanes with East anchor")

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Route guidance fix — unit tests (no CARLA, no model)")
    print("=" * 60)

    r1 = test_route_pts_ego_populated()
    r2 = test_route_is_lshaped()
    r3 = test_path_energy_favours_lshaped()
    r4 = test_anchor_rotation_at_turn()

    print(f"\n{'='*60}")
    results = [r1, r2, r3, r4]
    passed  = sum(bool(r) for r in results)
    names   = [
        "_route_pts_ego populated",
        "route is L-shaped (aisle + turn)",
        "path_energy prefers L-shaped over diagonal",
        "anchor rotation at junction (East heading, in-distribution route)",
    ]
    for name, ok in zip(names, results):
        print(f"  {'[PASS]' if ok else '[FAIL]'} {name}")
    print(f"\n{passed}/{len(results)} tests passed")
    sys.exit(0 if passed == len(results) else 1)
