"""
Unit tests for obb_aabb_overlap (runner_test_medium.py).

This is the SAT-based OBB-AABB collision check used for walker/cone
collisions — NOT the CARLA sensor-based 'actual_collisions'. It is
the ground-truth per-actor check that fires before the grid-quantized
collision_mask path, so correctness here directly affects the
walker_collisions_ref counter.

The function is inlined to keep the test self-contained and avoid
importing the full runner with its CARLA/srunner dependencies.
"""

import numpy as np
import pytest


# ── inlined from runner_test_medium.py ──────────────────────────────────────

def obb_aabb_overlap(cur, front_m, rear_m, half_w, aabb):
    """Exact OBB-AABB overlap via Separating Axis Theorem."""
    wx  = (aabb[0] + aabb[2]) / 2
    wy  = (aabb[1] + aabb[3]) / 2
    wdx = (aabb[2] - aabb[0]) / 2
    wdy = (aabb[3] - aabb[1]) / 2
    cos_a, sin_a = np.cos(cur.angle), np.sin(cur.angle)
    half_len = (front_m + rear_m) / 2
    obb_cx = cur.x + (front_m - rear_m) / 2 * cos_a
    obb_cy = cur.y + (front_m - rear_m) / 2 * sin_a
    dx, dy = wx - obb_cx, wy - obb_cy
    for ax, ay in [(1, 0), (0, 1), (cos_a, sin_a), (-sin_a, cos_a)]:
        obb_proj  = abs(half_len * (ax*cos_a + ay*sin_a)) + abs(half_w * (-ax*sin_a + ay*cos_a))
        aabb_proj = abs(wdx * ax) + abs(wdy * ay)
        if abs(dx*ax + dy*ay) > obb_proj + aabb_proj:
            return False
    return True


class Cur:
    """Minimal stand-in for TrajectoryPoint — only fields used by obb_aabb_overlap."""
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle  # radians


# Car dimensions matching the Lincoln MKZ default in v2.py
FRONT_M = 3.856
REAR_M  = 1.045
HALF_W  = 1.09


# ── axis-aligned (angle = 0) ─────────────────────────────────────────────────

def test_full_overlap_axis_aligned():
    """Walker AABB centred directly in front of a forward-facing car."""
    cur = Cur(0, 0, 0)
    # Walker centred at x=2, y=0 — well inside the car footprint
    aabb = [1.0, -0.5, 3.0, 0.5]
    assert obb_aabb_overlap(cur, FRONT_M, REAR_M, HALF_W, aabb) is True


def test_no_overlap_front_separation():
    """Walker is far ahead — clear gap on the longitudinal axis."""
    cur = Cur(0, 0, 0)
    aabb = [10.0, -0.5, 12.0, 0.5]
    assert obb_aabb_overlap(cur, FRONT_M, REAR_M, HALF_W, aabb) is False


def test_no_overlap_lateral_separation():
    """Walker is off to the side — clear gap on the lateral axis."""
    cur = Cur(0, 0, 0)
    aabb = [0.0, 3.0, 2.0, 5.0]
    assert obb_aabb_overlap(cur, FRONT_M, REAR_M, HALF_W, aabb) is False


def test_no_overlap_behind_car():
    """Walker is behind the rear of the car."""
    cur = Cur(0, 0, 0)
    # rear_m = 1.045, so rear edge is at x = -(1.045)
    aabb = [-5.0, -0.5, -2.0, 0.5]
    assert obb_aabb_overlap(cur, FRONT_M, REAR_M, HALF_W, aabb) is False


def test_touching_front_edge_is_overlap():
    """Walker AABB exactly touches the front face of the car — SAT uses >, so touching is overlap."""
    cur = Cur(0, 0, 0)
    # OBB front is at x = front_m = 3.856 from reference.
    # Walker left edge exactly at 3.856 → gap == projection sum → not strictly greater → overlap.
    aabb = [FRONT_M, -0.5, FRONT_M + 1.0, 0.5]
    assert obb_aabb_overlap(cur, FRONT_M, REAR_M, HALF_W, aabb) is True


def test_just_past_front_edge_no_overlap():
    """Walker AABB starts 1 mm past the front face — should be False."""
    cur = Cur(0, 0, 0)
    aabb = [FRONT_M + 0.001, -0.5, FRONT_M + 1.0, 0.5]
    assert obb_aabb_overlap(cur, FRONT_M, REAR_M, HALF_W, aabb) is False


def test_touching_lateral_edge_is_overlap():
    """Walker touches the side of the car exactly."""
    cur = Cur(0, 0, 0)
    aabb = [0.0, HALF_W, 2.0, HALF_W + 1.0]
    assert obb_aabb_overlap(cur, FRONT_M, REAR_M, HALF_W, aabb) is True


def test_just_past_lateral_edge_no_overlap():
    """Walker is 1 mm outside the lateral edge."""
    cur = Cur(0, 0, 0)
    aabb = [0.0, HALF_W + 0.001, 2.0, HALF_W + 1.0]
    assert obb_aabb_overlap(cur, FRONT_M, REAR_M, HALF_W, aabb) is False


# ── front/rear asymmetry ─────────────────────────────────────────────────────

def test_obb_center_accounts_for_asymmetry():
    """
    The OBB centre is offset from the reference point by (front_m - rear_m)/2
    along the heading. With front_m=4, rear_m=0 (symmetric about reference
    displaced to rear), the OBB extends [0, 4] from cur.x. A walker at x=[-1, -0.1]
    must NOT overlap.
    """
    cur = Cur(0, 0, 0)
    aabb = [-1.0, -0.5, -0.001, 0.5]
    assert obb_aabb_overlap(cur, 4.0, 0.0, 1.0, aabb) is False


def test_rear_extent_correct():
    """Walker immediately behind the rear edge should NOT overlap."""
    cur = Cur(0, 0, 0)
    rear = REAR_M
    aabb = [-rear - 1.0, -0.5, -rear - 0.001, 0.5]
    assert obb_aabb_overlap(cur, FRONT_M, REAR_M, HALF_W, aabb) is False


def test_rear_extent_just_inside():
    """Walker touching the rear edge from behind IS an overlap."""
    cur = Cur(0, 0, 0)
    rear = REAR_M
    aabb = [-rear - 1.0, -0.5, -rear, 0.5]
    assert obb_aabb_overlap(cur, FRONT_M, REAR_M, HALF_W, aabb) is True


# ── rotated car ──────────────────────────────────────────────────────────────

def test_rotated_90_overlap_ahead():
    """Car facing straight up (+y). Walker directly ahead in y."""
    cur = Cur(0, 0, np.pi / 2)
    # With angle=π/2: forward is +y. OBB centre at y=(front_m-rear_m)/2 ≈ 1.4
    # Walker at y=[3, 4] is ahead but separated.
    # Walker at y=[0.5, 1.5] is inside.
    aabb = [-0.5, 0.5, 0.5, 1.5]
    assert obb_aabb_overlap(cur, FRONT_M, REAR_M, HALF_W, aabb) is True


def test_rotated_90_no_overlap_to_side():
    """Car facing straight up. Walker clearly to the side (+x)."""
    cur = Cur(0, 0, np.pi / 2)
    aabb = [3.0, 0.0, 5.0, 2.0]
    assert obb_aabb_overlap(cur, FRONT_M, REAR_M, HALF_W, aabb) is False


def test_rotated_45_diagonal_separation():
    """
    Classic SAT test: AABB of the 45°-rotated car overlaps the walker's AABB,
    but the OBBs themselves are separated. obb_aabb_overlap must return False
    where a naive AABB-AABB test would wrongly return True.

    Car: angle=π/4, at origin. The car AABB extends ≈[-3.5, -3.5] to [3.5, 3.5].
    Walker at x=[3, 4], y=[-1, 0]: sits inside the car's AABB but is past the
    car's lateral boundary in the OBB frame.
    """
    cur = Cur(0, 0, np.pi / 4)
    aabb = [3.0, -1.0, 4.0, 0.0]
    assert obb_aabb_overlap(cur, FRONT_M, REAR_M, HALF_W, aabb) is False


def test_rotated_45_corner_overlap():
    """Car at 45°, walker at the diagonal tip of the car — should overlap."""
    cur = Cur(0, 0, np.pi / 4)
    # Forward direction: (0.707, 0.707). OBB centre at (1.06, 1.06).
    # Car corner at approx (1.06 + 2.5*0.707 - 1.09*0.707, 1.06 + 2.5*0.707 + 1.09*0.707)
    # ≈ (1.83, 3.37). Put a small walker right there.
    cx = 1.06 + 2.5*np.cos(np.pi/4) - HALF_W*np.sin(np.pi/4)
    cy = 1.06 + 2.5*np.sin(np.pi/4) + HALF_W*np.cos(np.pi/4)
    aabb = [cx - 0.3, cy - 0.3, cx + 0.3, cy + 0.3]
    assert obb_aabb_overlap(cur, FRONT_M, REAR_M, HALF_W, aabb) is True


# ── small / large walkers ────────────────────────────────────────────────────

def test_tiny_walker_inside_car():
    """Near-point walker well inside the car footprint."""
    cur = Cur(0, 0, 0)
    # Walker is a tiny 1cm x 1cm box at x=1, y=0
    aabb = [0.995, -0.005, 1.005, 0.005]
    assert obb_aabb_overlap(cur, FRONT_M, REAR_M, HALF_W, aabb) is True


def test_large_walker_fully_contains_car():
    """Walker bounding box completely wraps the car — always overlaps."""
    cur = Cur(0, 0, 0)
    aabb = [-20.0, -20.0, 20.0, 20.0]
    assert obb_aabb_overlap(cur, FRONT_M, REAR_M, HALF_W, aabb) is True


# ── non-zero car position ────────────────────────────────────────────────────

def test_car_offset_from_origin_overlap():
    """Car reference point not at origin — overlap should still be detected."""
    cur = Cur(10, 20, 0)
    aabb = [11.0, 19.5, 13.0, 20.5]
    assert obb_aabb_overlap(cur, FRONT_M, REAR_M, HALF_W, aabb) is True


def test_car_offset_from_origin_no_overlap():
    """Car not at origin, walker clearly separated."""
    cur = Cur(10, 20, 0)
    aabb = [0.0, 0.0, 1.0, 1.0]
    assert obb_aabb_overlap(cur, FRONT_M, REAR_M, HALF_W, aabb) is False
