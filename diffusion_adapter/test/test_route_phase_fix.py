"""
test_route_phase_fix.py
-----------------------
Verifies that _make_straight_route_lanes produces a bearing-aligned route.

With the bearing anchor always active, _make_straight_route_lanes always
produces a straight route from ego toward the destination.  In the bearing
anchor ego frame, this means the destination is at (dist_to_dest, ~0) and
the entire route has ego_y ≈ 0.

Tests:
  1. Bearing anchor phase — route is straight (ego_y ≈ 0) in bearing frame.
  2. Route includes a waypoint near the junction row (for both phases).
  3. Destination is straight ahead (ego_y ≈ 0) in bearing frame.
  4. Both phases produce valid tensor shapes (route_num, route_len, 12).
  5. Route reaches past the destination (model can overshoot safely).
"""

import sys, os
import numpy as np
import pytest

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

# Scenario constants
CAR_STD_AISLE   = np.array([285.0, -243.0])   # in aisle, 11m south of junction
CAR_STD_JUNC    = np.array([285.0, -235.73])   # 3m before junction (THRESHOLD=3.0)
JUNCTION_Y      = -232.0                        # approximate junction row y

DEST_STD_LEFT   = np.array([278.0, -232.0])    # left-side spot (West)
DEST_ANGLE_LEFT = float(np.pi)                 # West-facing spot

DEST_STD_RIGHT  = np.array([293.0, -232.0])    # right-side spot (East)
DEST_ANGLE_RIGHT = 0.0                         # East-facing spot

ROUTE_NUM = 10
ROUTE_LEN = 20


def _set_adapter_state(dest_std, dest_angle, suppress):
    da._dest_std          = dest_std.copy()
    da._dest_angle_std    = dest_angle
    da._suppress_guidance = suppress
    class _Cfg:
        route_num = ROUTE_NUM
        route_len = ROUTE_LEN
    da._config = _Cfg()
    da._route_pts_ego = None


def _bearing(car, dest):
    dx, dy = dest[0]-car[0], dest[1]-car[1]
    return float(np.arctan2(dy, dx))


# ===========================================================================
# Test 1: Aisle phase — initial segment of route is straight (before junction)
# ===========================================================================

@pytest.mark.parametrize("dest_std,dest_angle,label", [
    (DEST_STD_LEFT,  DEST_ANGLE_LEFT,  "left-spot"),
    (DEST_STD_RIGHT, DEST_ANGLE_RIGHT, "right-spot"),
])
def test_aisle_route_initial_segment_is_straight(dest_std, dest_angle, label):
    """
    With the bearing anchor always active, the route is a straight line from
    ego to destination.  In the bearing ego frame the destination is always
    straight ahead, so ego_y must be ≈ 0 throughout the entire route.
    """
    # Use bearing anchor (heading = bearing to destination)
    bearing = _bearing(CAR_STD_AISLE, dest_std)
    anchor  = np.array([CAR_STD_AISLE[0], CAR_STD_AISLE[1], bearing])
    _set_adapter_state(dest_std, dest_angle, suppress=True)

    da._make_straight_route_lanes(anchor)
    route_pts = da._route_pts_ego   # (N, 2) bearing-anchor ego frame

    assert route_pts is not None and len(route_pts) > 0

    # In the bearing anchor frame the entire route should have ego_y ≈ 0
    max_lateral = np.max(np.abs(route_pts[:, 1]))
    print(f"\n[{label} bearing] max|ego_y| = {max_lateral:.3f}m")
    assert max_lateral < 0.5, \
        f"[{label}] Route lateral in bearing frame = {max_lateral:.3f}m (should be < 0.5m)"


# ===========================================================================
# Test 2: Aisle phase — route has a waypoint near the junction row
# ===========================================================================

@pytest.mark.parametrize("dest_std,dest_angle,label", [
    (DEST_STD_LEFT,  DEST_ANGLE_LEFT,  "left-spot"),
    (DEST_STD_RIGHT, DEST_ANGLE_RIGHT, "right-spot"),
])
def test_aisle_route_reaches_junction(dest_std, dest_angle, label):
    """
    The route in aisle phase must reach the junction row (within 2m), confirming
    the aisle_align waypoint is present.  This is the key signal the model needs
    to go North first instead of immediately following its East prior.
    """
    heading = np.pi / 2
    anchor  = np.array([CAR_STD_AISLE[0], CAR_STD_AISLE[1], heading])
    _set_adapter_state(dest_std, dest_angle, suppress=True)

    da._make_straight_route_lanes(anchor)
    route_pts = da._route_pts_ego

    dist_to_junction = abs(JUNCTION_Y - CAR_STD_AISLE[1])  # ≈ 11m
    max_forward = np.max(route_pts[:, 0])
    print(f"\n[{label} aisle] dist_to_junction={dist_to_junction:.1f}m  "
          f"max route ego_x={max_forward:.1f}m")

    assert max_forward >= dist_to_junction - 2.0, \
        f"[{label}] Route only reaches {max_forward:.1f}m but junction is {dist_to_junction:.1f}m away"


# ===========================================================================
# Test 3: Junction phase — destination is straight ahead (ego_y ≈ 0)
# ===========================================================================

@pytest.mark.parametrize("dest_std,dest_angle,label", [
    (DEST_STD_LEFT,  DEST_ANGLE_LEFT,  "left-spot"),
    (DEST_STD_RIGHT, DEST_ANGLE_RIGHT, "right-spot"),
])
def test_junction_route_through_destination(dest_std, dest_angle, label):
    """
    In junction phase, the bearing anchor makes the destination straight ahead.
    The route goes through the destination, so the route is also straight in
    ego frame (ego_y ≈ 0 throughout).
    """
    car = CAR_STD_JUNC
    bearing = _bearing(car, dest_std)
    anchor  = np.array([car[0], car[1], bearing])
    _set_adapter_state(dest_std, dest_angle, suppress=False)

    da._make_straight_route_lanes(anchor)
    route_pts = da._route_pts_ego

    dist_to_dest = float(np.linalg.norm(dest_std - car))
    max_forward  = np.max(route_pts[:, 0])
    max_lateral  = np.max(np.abs(route_pts[:, 1]))

    print(f"\n[{label} junction] bearing={np.rad2deg(bearing):.1f}°  "
          f"dist={dist_to_dest:.1f}m  max|ego_y|={max_lateral:.3f}m  max_x={max_forward:.1f}m")

    assert max_forward >= dist_to_dest - 0.5, \
        f"[{label}] Route doesn't reach destination distance {dist_to_dest:.1f}m (max={max_forward:.1f})"
    assert max_lateral < 1.0, \
        f"[{label}] Junction route has lateral {max_lateral:.3f}m — dest should be straight ahead"


# ===========================================================================
# Test 4: Both phases produce valid tensor shapes
# ===========================================================================

@pytest.mark.parametrize("suppress,label", [(True, "aisle"), (False, "junction")])
def test_route_tensor_shape(suppress, label):
    car    = CAR_STD_AISLE if suppress else CAR_STD_JUNC
    h      = np.pi/2 if suppress else _bearing(CAR_STD_JUNC, DEST_STD_LEFT)
    anchor = np.array([car[0], car[1], h])
    _set_adapter_state(DEST_STD_LEFT, DEST_ANGLE_LEFT, suppress=suppress)

    lanes = da._make_straight_route_lanes(anchor)
    assert lanes.shape == (ROUTE_NUM, ROUTE_LEN, 12), \
        f"[{label}] Expected ({ROUTE_NUM},{ROUTE_LEN},12), got {lanes.shape}"


# ===========================================================================
# Test 5: Junction route extends well past destination
# ===========================================================================

@pytest.mark.parametrize("dest_std,dest_angle,label", [
    (DEST_STD_LEFT,  DEST_ANGLE_LEFT,  "left-spot"),
    (DEST_STD_RIGHT, DEST_ANGLE_RIGHT, "right-spot"),
])
def test_junction_route_has_overshoot(dest_std, dest_angle, label):
    """Route extends significantly past the destination so the model can 'see' past it."""
    car = CAR_STD_JUNC
    bearing = _bearing(car, dest_std)
    anchor  = np.array([car[0], car[1], bearing])
    _set_adapter_state(dest_std, dest_angle, suppress=False)

    da._make_straight_route_lanes(anchor)
    route_pts = da._route_pts_ego

    dist_to_dest = float(np.linalg.norm(dest_std - car))
    max_forward  = np.max(route_pts[:, 0])
    overshoot    = max_forward - dist_to_dest

    print(f"\n[{label} junction] dist_to_dest={dist_to_dest:.1f}m  overshoot={overshoot:.1f}m")
    assert overshoot >= 5.0, \
        f"[{label}] Route only extends {overshoot:.1f}m past dest — need ≥5m overshoot"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
