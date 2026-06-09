"""
Unit tests for ObstacleMap.generate_collision_mask() and obstacle_map_from_bbs().

generate_collision_mask is used in runner_test_medium to build the car footprint
grid for both the cone collision check (grid-based) and the near_miss calculation.
Correctness here ensures the car "shadow" is in the right place and orientation.

Both classes are inlined (minimal versions) to avoid importing v2.py's full
dependency chain (CARLA, srunner, pykalman, etc.).
"""

import numpy as np
import pytest


RESOLUTION = 0.25   # metres per grid cell, must match v2.py


# ── minimal ObstacleMap inlined from v2.py ───────────────────────────────────

class ObstacleMap:
    def __init__(self, min_x, min_y, obs):
        self.min_x = min_x
        self.min_y = min_y
        self.obs = obs

    def transform_coord(self, x, y):
        return int((x - self.min_x) / RESOLUTION), int((y - self.min_y) / RESOLUTION)

    def generate_collision_mask(self, wp, front_m=3.856, rear_m=1.045, half_width_m=1.09):
        mask = np.zeros_like(self.obs, dtype=bool)
        x_coords, y_coords = np.meshgrid(
            np.arange(self.obs.shape[0]), np.arange(self.obs.shape[1])
        )
        rear_px  = rear_m       / RESOLUTION
        front_px = front_m      / RESOLUTION
        width_px = half_width_m * 2 / RESOLUTION

        cx, cy = self.transform_coord(wp.x, wp.y)
        R = np.array([
            [np.cos(-wp.angle), -np.sin(-wp.angle)],
            [np.sin(-wp.angle),  np.cos(-wp.angle)],
        ])
        coords  = np.stack([x_coords.flatten() - cx, y_coords.flatten() - cy])
        rotated = R @ coords
        hits = (
            (rotated[0] >= -rear_px) & (rotated[0] <= front_px)
            & (rotated[1] >= -width_px / 2) & (rotated[1] <= width_px / 2)
        )
        mask |= hits.reshape(self.obs.shape[1], self.obs.shape[0]).T
        return mask


def obstacle_map_from_bbs(bbs, min_x=None, min_y=None, size=200):
    """Simplified version: creates an all-zero map big enough for tests."""
    if min_x is None:
        min_x = -size / 2 * RESOLUTION
    if min_y is None:
        min_y = -size / 2 * RESOLUTION
    obs = np.zeros((size, size), dtype=int)
    return ObstacleMap(min_x, min_y, obs)


class WP:
    """Minimal TrajectoryPoint stand-in."""
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle  # radians


# ── generate_collision_mask ───────────────────────────────────────────────────

def test_car_centre_pixel_in_mask_angle_zero():
    """The grid cell containing the car reference point is always inside the footprint."""
    om = obstacle_map_from_bbs([], min_x=-25.0, min_y=-25.0, size=200)
    wp = WP(0.0, 0.0, 0.0)
    mask = om.generate_collision_mask(wp, front_m=4.0, rear_m=1.0, half_width_m=1.0)
    cx, cy = om.transform_coord(0.0, 0.0)
    assert mask[cx, cy], "Reference pixel not in collision mask"


def test_mask_nonzero():
    """Mask must cover a non-trivial area for a normal car."""
    om = obstacle_map_from_bbs([], min_x=-25.0, min_y=-25.0, size=200)
    mask = om.generate_collision_mask(WP(0, 0, 0), front_m=4.0, rear_m=1.0, half_width_m=1.0)
    assert mask.sum() > 0


def test_mask_axis_aligned_shape_longitudinal():
    """
    At angle=0 the mask footprint should extend front_m forward and rear_m back
    from the reference pixel along the x-axis.
    """
    front_m, rear_m, half_w = 4.0, 1.0, 1.0
    om = obstacle_map_from_bbs([], min_x=-25.0, min_y=-25.0, size=200)
    wp = WP(0.0, 0.0, 0.0)
    mask = om.generate_collision_mask(wp, front_m=front_m, rear_m=rear_m, half_width_m=half_w)

    cx, cy = om.transform_coord(0.0, 0.0)
    # A pixel exactly at front_m along x should be inside
    fx, fy = om.transform_coord(front_m - RESOLUTION, 0.0)
    assert mask[fx, fy], "Front of car not covered by mask"

    # A pixel one step beyond front_m should be outside
    ox, oy = om.transform_coord(front_m + RESOLUTION, 0.0)
    if 0 <= ox < mask.shape[0] and 0 <= oy < mask.shape[1]:
        assert not mask[ox, oy], "Pixel past front face is in mask"

    # A pixel exactly at rear_m behind should be inside
    rx, ry = om.transform_coord(-(rear_m - RESOLUTION), 0.0)
    assert mask[rx, ry], "Rear of car not covered by mask"

    # A pixel one step beyond rear should be outside
    bx, by = om.transform_coord(-(rear_m + RESOLUTION), 0.0)
    if 0 <= bx < mask.shape[0] and 0 <= by < mask.shape[1]:
        assert not mask[bx, by], "Pixel past rear face is in mask"


def test_mask_axis_aligned_shape_lateral():
    """At angle=0, mask should not exceed half_width_m on either side."""
    front_m, rear_m, half_w = 4.0, 1.0, 1.0
    om = obstacle_map_from_bbs([], min_x=-25.0, min_y=-25.0, size=200)
    wp = WP(0.0, 0.0, 0.0)
    mask = om.generate_collision_mask(wp, front_m=front_m, rear_m=rear_m, half_width_m=half_w)

    # Centre of car, just inside the lateral edge
    ix, iy = om.transform_coord(1.0, half_w - RESOLUTION)
    assert mask[ix, iy], "Interior lateral pixel not in mask"

    # One step outside
    ox, oy = om.transform_coord(1.0, half_w + RESOLUTION)
    if 0 <= ox < mask.shape[0] and 0 <= oy < mask.shape[1]:
        assert not mask[ox, oy], "Pixel outside lateral edge is in mask"


def test_mask_rotated_90_shifts_to_y_axis():
    """
    At angle=π/2 (car faces +y), the longitudinal axis becomes y.
    The pixel front_m ahead of the reference along +y must be in the mask,
    and the same distance along +x must not.
    """
    front_m, rear_m, half_w = 4.0, 1.0, 1.0
    om = obstacle_map_from_bbs([], min_x=-25.0, min_y=-25.0, size=200)
    wp = WP(0.0, 0.0, np.pi / 2)
    mask = om.generate_collision_mask(wp, front_m=front_m, rear_m=rear_m, half_width_m=half_w)

    # front along +y
    fy_x, fy_y = om.transform_coord(0.0, front_m - RESOLUTION)
    assert mask[fy_x, fy_y], "Forward pixel along +y axis not in mask (rotated 90°)"

    # same distance along +x — should be outside
    fx_x, fx_y = om.transform_coord(front_m - RESOLUTION, 0.0)
    if 0 <= fx_x < mask.shape[0] and 0 <= fx_y < mask.shape[1]:
        assert not mask[fx_x, fx_y], "Lateral pixel incorrectly in mask (rotated 90°)"


def test_mask_car_outside_map_does_not_crash():
    """Car reference at map edge — mask computation must not raise."""
    om = obstacle_map_from_bbs([], min_x=0.0, min_y=0.0, size=50)
    wp = WP(0.0, 0.0, 0.0)   # at map origin — rear hangs off the edge
    mask = om.generate_collision_mask(wp, front_m=4.0, rear_m=1.0, half_width_m=1.0)
    assert mask is not None


def test_mask_dimensions_match_obs():
    """Returned mask must have the same shape as obs."""
    om = obstacle_map_from_bbs([], min_x=-25.0, min_y=-25.0, size=200)
    mask = om.generate_collision_mask(WP(0, 0, 0))
    assert mask.shape == om.obs.shape


# ── obstacle_map_from_bbs (testbed version) ──────────────────────────────────

def test_obstacle_map_from_bbs_creates_map():
    om = obstacle_map_from_bbs([], min_x=-10.0, min_y=-10.0, size=80)
    assert om.obs.shape == (80, 80)
    assert om.min_x == -10.0
    assert om.min_y == -10.0


def test_transform_coord_roundtrip():
    om = obstacle_map_from_bbs([], min_x=-10.0, min_y=-10.0, size=80)
    wx, wy = 3.5, -2.0
    gx, gy = om.transform_coord(wx, wy)
    rx = gx * RESOLUTION + om.min_x
    ry = gy * RESOLUTION + om.min_y
    # Roundtrip within one pixel (floor in transform_coord)
    assert abs(rx - wx) < RESOLUTION
    assert abs(ry - wy) < RESOLUTION


# ── collision_tracker map-bounds gap (gap #3) ─────────────────────────────────
#
# collision_tracker is built ONCE from initial actor positions. If car.cur
# moves outside the grid (e.g., actor at destination far from initial actors),
# generate_collision_mask returns an all-False mask silently — near-miss and
# cone checks go dark for the rest of the run with no error.
#
# These tests document the failure mode and the invariant production must satisfy.
# ─────────────────────────────────────────────────────────────────────────────

def test_collision_mask_empty_when_car_outside_grid():
    """
    If car.cur is outside the collision_tracker grid, generate_collision_mask
    returns an all-False mask — the car footprint is silently off-grid.

    This demonstrates the failure mode that occurs when collision_tracker was
    built from actors that don't extend to the ego's current position.
    """
    # Small grid: covers world coords [0, 12.5] x [0, 12.5]
    om = obstacle_map_from_bbs([], min_x=0.0, min_y=0.0, size=50)

    # Car positioned well outside the grid
    wp_outside = WP(x=30.0, y=30.0, angle=0.0)
    mask = om.generate_collision_mask(wp_outside, front_m=4.0, rear_m=1.0, half_width_m=1.0)

    assert mask.sum() == 0, (
        "Car outside grid returns non-empty mask — unexpected. "
        "Expected: all-False (silent off-grid failure mode)."
    )


def test_collision_mask_nonempty_when_car_inside_grid():
    """
    Counterpart: car.cur well inside the grid must produce a non-empty mask.
    This is the invariant production relies on every tick.
    """
    om = obstacle_map_from_bbs([], min_x=-25.0, min_y=-25.0, size=200)
    wp = WP(x=0.0, y=0.0, angle=0.0)
    mask = om.generate_collision_mask(wp, front_m=4.0, rear_m=1.0, half_width_m=1.0)
    assert mask.sum() > 0, "Car inside grid must produce a non-empty collision mask"


def test_collision_tracker_must_cover_all_ego_waypoints():
    """
    Production builds collision_tracker from initial actor positions with 10 m padding.
    If the ego drives more than ~10 m from the nearest initial actor, the tracker
    grid does not cover the ego and generate_collision_mask returns empty.

    Simulates: initial actors clustered at origin (±5 m), ego drives to x=20 m.
    The tracker covers [-15, 15] in x (5 + 10 m padding). At x=20 the mask is empty.
    """
    # Actors span x=[−5, 5], y=[−5, 5] → tracker covers x=[−15, 15], y=[−15, 15]
    # (the simplified obstacle_map_from_bbs in this test file takes explicit min_x/size)
    tracker_min_x = -5.0 - 10.0  # = -15 m
    tracker_size  = int((30.0 + 1) / RESOLUTION)  # 30 m / 0.25 = 120 cells
    om = obstacle_map_from_bbs([], min_x=tracker_min_x, min_y=-15.0, size=tracker_size)

    # Ego at destination 20 m away — OUTSIDE the tracker
    wp_dest = WP(x=20.0, y=0.0, angle=0.0)
    mask_dest = om.generate_collision_mask(wp_dest, front_m=4.0, rear_m=1.0, half_width_m=1.0)

    # Ego at start 0 m — INSIDE the tracker
    wp_start = WP(x=0.0, y=0.0, angle=0.0)
    mask_start = om.generate_collision_mask(wp_start, front_m=4.0, rear_m=1.0, half_width_m=1.0)

    assert mask_start.sum() > 0, "Ego at start must be covered by collision_tracker"
    assert mask_dest.sum() == 0, (
        "Ego at destination (outside tracker bounds) silently returns empty mask. "
        "Near-miss/cone checks go dark — production must ensure tracker covers full path."
    )
