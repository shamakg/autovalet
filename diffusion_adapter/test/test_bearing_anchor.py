"""
test_bearing_anchor.py
----------------------
Targeted unit test for the bearing-to-destination anchor fix.

ROOT CAUSE (what this test addresses):

  After the aisle-phase suppression fix, anchor rotation correctly triggers at
  the junction.  However, effective_h = _dest_angle_std = 0.0 (exactly East).

  When the car is 0.65 m south of the destination row (car y=-233.38, dest
  y=-232.73), with effective_h=0.0 (East), the destination appears at:
      dest_ego = (7.76, +0.65)   — 0.65 m left (North)

  The model's South-drift prior (-10 m in 8 s) overwhelms the +0.65 m North
  signal.  goal_energy's gradient at the final waypoint tilts the trajectory
  South, so pred[-1] lands at ego=(35.76, -9.97).  Pure pursuit lookahead
  ends up 4 m south of the junction row, and the car's arc hits parked cars.

FIX: use the actual bearing from car position to destination as effective_h:
      effective_h = atan2(dest_std_y - std_y, dest_std_x - std_x)

  At the trigger position: effective_h = atan2(0.65, 7.76) ≈ 0.0836 rad ≈ 4.8°
  With this heading, the destination appears at ego=(7.82, 0.0) — exactly ahead.
  goal_energy gradient now points purely forward (+x), aligning with the model
  prior, so the trajectory stays on the destination row.

Tests:
  1. With exact-East heading (broken): dest appears at (7.76, +0.65)    [diagnose]
  2. With bearing heading (fix): dest appears at (7.82, 0.0)            [verify fix]
  3. Goal-energy gradient with exact-East: has lateral component         [diagnose]
  4. Goal-energy gradient with bearing: lateral component ≈ 0           [verify fix]
  5. Bearing formula is robust at several lateral offsets                [edge cases]

No model, no CARLA server, no GPU required.
"""

import sys, os
import numpy as np
import torch
import pytest

# ---------------------------------------------------------------------------
# Path setup — mirrors the layout in test_guidance_suppression.py
# ---------------------------------------------------------------------------
_HERE          = os.path.dirname(os.path.abspath(__file__))
_ADAPTER_ROOT  = os.path.dirname(_HERE)
_AUTOVALET     = os.path.dirname(_ADAPTER_ROOT)
_NUPLAN_ROOT   = os.path.join(_ADAPTER_ROOT, "nuplan-devkit")
_DP_ROOT       = os.path.join(_ADAPTER_ROOT, "Diffusion-Planner")

_CARLA_ROOT    = "/home/sumesh/opt/carla/PythonAPI/carla"
_SCENARIO_ROOT = "/home/sumesh/carla_garage/scenario_runner"
_LB_ROOT       = "/home/sumesh/carla_garage/leaderboard"

for _p in [_CARLA_ROOT, _SCENARIO_ROOT, _LB_ROOT,
           _HERE, _AUTOVALET, _ADAPTER_ROOT, _NUPLAN_ROOT, _DP_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from diffusion_planner.data_process.utils import _state_se2_array_to_transform_matrix

# ---------------------------------------------------------------------------
# Constants (from diff_adapter.py and Diffusion-Planner)
# ---------------------------------------------------------------------------
MEAN_XY    = np.array([10.0, 0.0])
STD_XY     = np.array([20.0, 20.0])
GOAL_SCALE = 5000.0

# CARLA scenario values from the logged junction trigger tick
DEST_STD   = np.array([293.22, -232.73])   # destination in standard coords
CAR_STD    = np.array([285.46, -233.38])   # car position at trigger tick
DEST_ANGLE = 0.0                            # _dest_angle_std = 0 (East)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dest_in_ego_frame(car_pos: np.ndarray, heading_rad: float,
                        dest: np.ndarray) -> tuple[float, float]:
    """Transform destination into ego frame given car position and heading."""
    anchor = np.array([car_pos[0], car_pos[1], heading_rad], dtype=np.float64)
    M      = _state_se2_array_to_transform_matrix(anchor)
    M_inv  = np.linalg.inv(M)
    dest_h = np.array([dest[0], dest[1], 1.0])
    ego    = M_inv @ dest_h
    return float(ego[0]), float(ego[1])


def _goal_energy_grad(final_real_xy: np.ndarray,
                       dest_ego_xy:   np.ndarray) -> tuple[float, np.ndarray]:
    """
    Compute goal_energy gradient w.r.t. the NORMALISED final waypoint.
    Returns (energy_value, grad_normalised_shape=(2,)).
    """
    sxy = torch.tensor(STD_XY,  dtype=torch.float32)
    mxy = torch.tensor(MEAN_XY, dtype=torch.float32)
    dst = torch.tensor(dest_ego_xy, dtype=torch.float32)

    # Convert real → normalised (the tensor that has grad)
    final_norm_np = (final_real_xy - MEAN_XY) / STD_XY
    fn = torch.tensor(final_norm_np, dtype=torch.float32, requires_grad=True)
    fn_real = fn * sxy + mxy
    goal_dist   = torch.norm(fn_real - dst)
    goal_energy = -(GOAL_SCALE / sxy[0]) * goal_dist
    goal_energy.backward()
    return goal_energy.item(), fn.grad.detach().numpy().copy()


def _bearing_heading(car_pos: np.ndarray, dest: np.ndarray) -> float:
    """Bearing from car to destination."""
    dx = dest[0] - car_pos[0]
    dy = dest[1] - car_pos[1]
    return float(np.arctan2(dy, dx))


# ===========================================================================
# Test 1: With exact-East heading, destination appears laterally offset
# ===========================================================================

def test_exact_east_dest_is_not_straight_ahead():
    """
    DIAGNOSE: with effective_h = 0.0 (exact East), destination has a non-trivial
    lateral component in ego frame.  The car is 0.65 m south of dest row.
    """
    ego_x, ego_y = _dest_in_ego_frame(CAR_STD, heading_rad=0.0, dest=DEST_STD)
    forward  = ego_x   # metres ahead  (+x in ego)
    lateral  = ego_y   # metres left   (+y in ego)
    lateral_abs = abs(lateral)

    print(f"\n[exact-East] dest_ego=({ego_x:.4f}, {ego_y:.4f}) m")
    print(f"  forward={forward:.4f} m,  |lateral|={lateral_abs:.4f} m")

    assert forward  > 7.0,  f"expected forward ≈ 7.76 m, got {forward:.3f}"
    assert lateral_abs > 0.3, f"expected lateral > 0.3 m (non-trivial), got {lateral_abs:.3f}"
    # Destination is slightly NORTH (positive y) because car is 0.65 m south of dest row
    assert lateral > 0.0,  f"expected positive (North) lateral, got {lateral:.4f}"


# ===========================================================================
# Test 2: With bearing heading, destination appears exactly straight ahead
# ===========================================================================

def test_bearing_heading_makes_dest_straight_ahead():
    """
    FIX: bearing heading = atan2(dest_y - car_y, dest_x - car_x) makes the
    destination appear at (dist, 0) in ego frame — exactly forward, zero lateral.
    """
    bearing = _bearing_heading(CAR_STD, DEST_STD)
    ego_x, ego_y = _dest_in_ego_frame(CAR_STD, heading_rad=bearing, dest=DEST_STD)

    print(f"\n[bearing] bearing={np.rad2deg(bearing):.4f}°")
    print(f"[bearing] dest_ego=({ego_x:.6f}, {ego_y:.6f}) m")

    dist_to_dest = float(np.linalg.norm(DEST_STD - CAR_STD))
    assert abs(ego_x - dist_to_dest) < 0.01, \
        f"ego_x={ego_x:.4f} should ≈ dist_to_dest={dist_to_dest:.4f}"
    assert abs(ego_y) < 0.001, \
        f"ego_y={ego_y:.6f} should be ≈ 0 (straight ahead)"


# ===========================================================================
# Test 3: goal_energy gradient with exact-East has large lateral component
# ===========================================================================

def test_goal_energy_grad_exact_east_has_lateral():
    """
    DIAGNOSE: with exact-East, goal_energy gradient at final waypoint has a
    large negative-y (South-pulling) component, tilting the trajectory south.
    """
    # Ego frame with exact-East heading: dest at (7.76, +0.65)
    dest_ego_x, dest_ego_y = _dest_in_ego_frame(CAR_STD, heading_rad=0.0, dest=DEST_STD)
    dest_ego = np.array([dest_ego_x, dest_ego_y])

    # Place final waypoint at a reasonable mid-point (15m ahead on dest row)
    final_real = np.array([15.0, 0.0])
    _, grad = _goal_energy_grad(final_real, dest_ego)

    grad_x, grad_y = grad
    print(f"\n[exact-East grad] grad_x={grad_x:.1f}, grad_y={grad_y:.1f}")
    print(f"  |grad_y| / |grad_x| = {abs(grad_y)/max(abs(grad_x),1e-6):.3f}")

    # Lateral (y) component should be significant
    assert abs(grad_y) > abs(grad_x) * 0.05, \
        f"Expected non-trivial lateral gradient, got grad=({grad_x:.1f},{grad_y:.1f})"


# ===========================================================================
# Test 4: goal_energy gradient with bearing heading — lateral ≈ 0
# ===========================================================================

def test_goal_energy_grad_bearing_is_purely_forward():
    """
    FIX: with bearing heading, destination is at (7.82, 0) in ego frame.
    goal_energy gradient at final waypoint should be almost purely forward/backward
    (no lateral component).  |grad_y| / |grad_x| should be < 0.02.

    Note on sign: when final_real is BEHIND the destination (e.g. 5m < 7.82m),
    gradient is positive-x (pulling forward toward dest).  When it is PAST the
    destination (e.g. 15m > 7.82m), gradient is negative-x (pulling backward).
    Either way the lateral ratio must be ≈ 0.
    """
    bearing = _bearing_heading(CAR_STD, DEST_STD)
    dest_ego_x, dest_ego_y = _dest_in_ego_frame(CAR_STD, heading_rad=bearing, dest=DEST_STD)
    dest_ego = np.array([dest_ego_x, dest_ego_y])

    # Place final waypoint SHORT of the destination so gradient points FORWARD (+x).
    # dest_ego_x ≈ 7.82 m, so final at 5 m is clearly behind it.
    final_real = np.array([5.0, 0.0])
    _, grad = _goal_energy_grad(final_real, dest_ego)

    grad_x, grad_y = grad
    lateral_ratio = abs(grad_y) / max(abs(grad_x), 1e-6)
    print(f"\n[bearing grad] dest_ego=({dest_ego_x:.4f},{dest_ego_y:.6f})")
    print(f"[bearing grad] grad_x={grad_x:.1f}, grad_y={grad_y:.1f}")
    print(f"  |grad_y| / |grad_x| = {lateral_ratio:.4f}")

    assert lateral_ratio < 0.02, \
        f"Expected near-zero lateral gradient, got ratio={lateral_ratio:.4f}"
    # Final is behind destination → gradient must pull forward
    assert grad_x > 10.0, \
        f"Expected forward (positive x) gradient, got grad_x={grad_x:.1f}"


# ===========================================================================
# Test 5: Bearing formula is robust at various lateral offsets
# ===========================================================================

@pytest.mark.parametrize("car_south_offset,expected_bearing_deg_range", [
    (0.0,  (-1.0, 1.0)),      # car exactly on dest row → bearing ≈ 0° East
    (0.65, (3.0, 7.0)),       # car 0.65 m south (actual scenario) → bearing ≈ 4.8°
    (2.0,  (13.0, 17.0)),     # car 2 m south → bearing ≈ 14.4°
    (5.0,  (30.0, 40.0)),     # car 5 m south → bearing ≈ 32.8°
])
def test_bearing_zero_lateral_across_offsets(car_south_offset, expected_bearing_deg_range):
    """
    At any car offset south of the dest row, bearing heading should produce
    dest_ego_y ≈ 0 (destination exactly ahead).
    """
    # CARLA south = decreasing y in std coords
    car_pos = np.array([CAR_STD[0], DEST_STD[1] - car_south_offset])
    bearing = _bearing_heading(car_pos, DEST_STD)
    bearing_deg = np.rad2deg(bearing)
    _, dest_ego_y = _dest_in_ego_frame(car_pos, heading_rad=bearing, dest=DEST_STD)

    lo, hi = expected_bearing_deg_range
    print(f"\n[offset={car_south_offset}m] bearing={bearing_deg:.2f}°  dest_ego_y={dest_ego_y:.6f}")

    assert lo <= bearing_deg <= hi, \
        f"bearing {bearing_deg:.2f}° not in [{lo},{hi}] for offset={car_south_offset}m"
    assert abs(dest_ego_y) < 0.001, \
        f"dest_ego_y={dest_ego_y:.6f} should be ≈ 0 for offset={car_south_offset}m"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
