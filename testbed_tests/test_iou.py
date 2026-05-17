"""
Unit tests for the IOU calculation in CarlaCar.iou() (v2.py).

The IOU is computed as:
  Polygon(rotated_car_vertices).intersection(Polygon(destination_aabb)).area
  / union

destination_bb = [x_min, y_min, x_max, y_max] (axis-aligned, from approximate_bb_from_center)
car_vertices = 4 corners of the bounding box rotated by the car's yaw.

The math is inlined here to avoid importing v2.py and its heavy dependency chain
(CARLA, srunner, pykalman, hybrid_a_star, etc.).
"""

import numpy as np
import pytest
from shapely.geometry import Polygon


# ── inlined from v2.py CarlaCar.iou() ───────────────────────────────────────

def compute_iou(car_cx, car_cy, car_yaw_rad, half_ext_x, half_ext_y,
                dest_bb):
    """
    car_cx, car_cy   : car centre in world coords
    car_yaw_rad      : car heading in radians (CARLA convention: clockwise positive,
                       but the math works the same — rotation is applied consistently)
    half_ext_x/y     : bounding box half-extents from CARLA's bounding_box.extent
    dest_bb          : [x_min, y_min, x_max, y_max] destination AABB
    """
    R = np.array([
        [np.cos(car_yaw_rad), -np.sin(car_yaw_rad)],
        [np.sin(car_yaw_rad),  np.cos(car_yaw_rad)]
    ])
    corners_local = [
        [-half_ext_x, -half_ext_y],
        [-half_ext_x,  half_ext_y],
        [ half_ext_x,  half_ext_y],
        [ half_ext_x, -half_ext_y],
    ]
    car_vertices = [
        np.dot(R, np.array(c)) + np.array([car_cx, car_cy])
        for c in corners_local
    ]
    destination_vertices = [
        (dest_bb[0], dest_bb[1]),
        (dest_bb[0], dest_bb[3]),
        (dest_bb[2], dest_bb[3]),
        (dest_bb[2], dest_bb[1]),
    ]
    car_poly  = Polygon(car_vertices)
    dest_poly = Polygon(destination_vertices)
    return car_poly.intersection(dest_poly).area / car_poly.union(dest_poly).area


# ── fixtures ─────────────────────────────────────────────────────────────────

# approximate_bb_from_center gives half-widths 2.4 (x) and 0.96 (y)
SPOT_HW_X = 2.4
SPOT_HW_Y = 0.96

# Lincoln MKZ bounding box extents (from CARLA)
CAR_HW_X = 2.4
CAR_HW_Y = 0.96


def spot_bb(cx, cy):
    return [cx - SPOT_HW_X, cy - SPOT_HW_Y, cx + SPOT_HW_X, cy + SPOT_HW_Y]


# ── perfect alignment ─────────────────────────────────────────────────────────

def test_perfect_park_axis_aligned():
    """Car exactly matches the destination spot at 0° yaw → IOU = 1.0."""
    cx, cy = 0.0, 0.0
    iou = compute_iou(cx, cy, 0.0, CAR_HW_X, CAR_HW_Y, spot_bb(cx, cy))
    assert abs(iou - 1.0) < 1e-9


def test_perfect_park_180_degrees():
    """180° yaw on an axis-aligned slot: car is still congruent → IOU = 1.0."""
    cx, cy = 5.0, -10.0
    iou = compute_iou(cx, cy, np.pi, CAR_HW_X, CAR_HW_Y, spot_bb(cx, cy))
    assert abs(iou - 1.0) < 1e-9


def test_perfect_park_arbitrary_position():
    """Different map location, still perfect alignment."""
    cx, cy = 285.0, -215.0
    iou = compute_iou(cx, cy, 0.0, CAR_HW_X, CAR_HW_Y, spot_bb(cx, cy))
    assert abs(iou - 1.0) < 1e-9


# ── no overlap ───────────────────────────────────────────────────────────────

def test_no_overlap_x():
    """Car driven away from spot — no intersection."""
    cx, cy = 0.0, 0.0
    bb = spot_bb(cx, cy)
    # Move car 20 m along x
    iou = compute_iou(20.0, cy, 0.0, CAR_HW_X, CAR_HW_Y, bb)
    assert iou == 0.0


def test_no_overlap_y():
    cx, cy = 0.0, 0.0
    bb = spot_bb(cx, cy)
    iou = compute_iou(cx, 20.0, 0.0, CAR_HW_X, CAR_HW_Y, bb)
    assert iou == 0.0


# ── partial overlap ───────────────────────────────────────────────────────────

def test_partial_overlap_x_offset():
    """Car slid halfway to the right — IOU is positive but < 1."""
    cx, cy = 0.0, 0.0
    bb = spot_bb(cx, cy)
    # Slide car right by its half-x extent: overlap is half the car width
    iou = compute_iou(CAR_HW_X, cy, 0.0, CAR_HW_X, CAR_HW_Y, bb)
    assert 0.0 < iou < 1.0


def test_partial_overlap_y_offset():
    cx, cy = 0.0, 0.0
    bb = spot_bb(cx, cy)
    iou = compute_iou(cx, CAR_HW_Y, 0.0, CAR_HW_X, CAR_HW_Y, bb)
    assert 0.0 < iou < 1.0


def test_small_offset_gives_high_iou():
    """0.1 m lateral offset in a 0.96 m half-width spot → IOU still > 0.9."""
    cx, cy = 0.0, 0.0
    bb = spot_bb(cx, cy)
    iou = compute_iou(cx, 0.1, 0.0, CAR_HW_X, CAR_HW_Y, bb)
    assert iou > 0.9


def test_large_offset_gives_low_iou():
    """Offset by 80% of half-width → IOU < 0.5."""
    cx, cy = 0.0, 0.0
    bb = spot_bb(cx, cy)
    iou = compute_iou(cx, CAR_HW_Y * 0.8, 0.0, CAR_HW_X, CAR_HW_Y, bb)
    assert iou < 0.5


# ── rotation ─────────────────────────────────────────────────────────────────

def test_small_rotation_still_high_iou():
    """5° yaw error in an axis-aligned spot → IOU is still substantial.
    The Lincoln MKZ is elongated (4.8m x 1.92m), so 5° swings the corners
    0.08 m into the slot margin, dropping IOU to ~0.89. Threshold is 0.85."""
    cx, cy = 0.0, 0.0
    iou = compute_iou(cx, cy, np.deg2rad(5), CAR_HW_X, CAR_HW_Y, spot_bb(cx, cy))
    assert iou > 0.85


def test_45_degree_rotation_reduces_iou():
    """45° misalignment in an axis-aligned spot → significantly reduced IOU."""
    cx, cy = 0.0, 0.0
    iou = compute_iou(cx, cy, np.deg2rad(45), CAR_HW_X, CAR_HW_Y, spot_bb(cx, cy))
    assert iou < 0.8


def test_90_degree_rotation():
    """
    90° rotation for an asymmetric car (ext_x != ext_y).
    The rotated car is taller-than-wide in the spot, so IOU < 1.
    With our dims (2.4 x 0.96), spot area = 9.216, car area = 9.216.
    At 90° the car extends 0.96 in x and 2.4 in y; the slot is 2.4 x 0.96.
    Intersection is 0.96 x 0.96 = 0.9216; union = 2*(9.216) - 0.9216 = 17.51.
    IOU = 0.9216 / 17.51 ≈ 0.053.
    """
    cx, cy = 0.0, 0.0
    iou = compute_iou(cx, cy, np.deg2rad(90), CAR_HW_X, CAR_HW_Y, spot_bb(cx, cy))
    expected = (CAR_HW_Y * CAR_HW_Y) / (CAR_HW_X * CAR_HW_Y + CAR_HW_X * CAR_HW_Y - CAR_HW_Y * CAR_HW_Y)
    assert abs(iou - expected) < 0.02   # allow small shapely rounding


def test_iou_symmetric_about_180():
    """IOU at +θ equals IOU at -θ (the polygon is centrosymmetric)."""
    cx, cy = 0.0, 0.0
    bb = spot_bb(cx, cy)
    theta = np.deg2rad(30)
    iou_pos = compute_iou(cx, cy,  theta, CAR_HW_X, CAR_HW_Y, bb)
    iou_neg = compute_iou(cx, cy, -theta, CAR_HW_X, CAR_HW_Y, bb)
    assert abs(iou_pos - iou_neg) < 1e-9


# ── monotonicity ─────────────────────────────────────────────────────────────

def test_iou_decreases_with_lateral_offset():
    """IOU must be monotonically non-increasing as we slide further from centre."""
    cx, cy = 0.0, 0.0
    bb = spot_bb(cx, cy)
    offsets = [0.0, 0.2, 0.5, 0.8, 1.2]
    ious = [compute_iou(cx, cy + d, 0.0, CAR_HW_X, CAR_HW_Y, bb) for d in offsets]
    for i in range(len(ious) - 1):
        assert ious[i] >= ious[i + 1], f"IOU increased at offset {offsets[i+1]}: {ious}"


def test_iou_bounded_zero_one():
    """IOU must always be in [0, 1]."""
    cx, cy = 0.0, 0.0
    bb = spot_bb(cx, cy)
    for yaw in np.linspace(0, np.pi, 10):
        for dy in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            iou = compute_iou(cx, cy + dy, yaw, CAR_HW_X, CAR_HW_Y, bb)
            assert 0.0 <= iou <= 1.0 + 1e-9, f"IOU out of range: {iou}"
