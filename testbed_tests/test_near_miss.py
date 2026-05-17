"""
Unit tests for the near_miss() function from testbed/v2_experiment_utils.py.

near_miss(collision_mask, ground_truth_obs) returns:
  - 0.0 if collision_mask and ground_truth_obs overlap
  - inf if no obstacle pixels exist
  - min_pixel_distance * 0.25 (metres) otherwise

IMPORTANT: The function samples car pixels with [::5] for performance.
Tests are designed so the nearest car pixel is always in the sampled set,
avoiding false failures from that approximation.
"""

import numpy as np
import pytest


# ── inlined from testbed/v2_experiment_utils.py ─────────────────────────────

def near_miss(collision_mask, ground_truth_obs):
    if np.any(collision_mask & (ground_truth_obs == 1)):
        return 0.0

    car_pixels = np.argwhere(collision_mask)
    obstacle_pixels = np.argwhere(ground_truth_obs == 1)

    if len(obstacle_pixels) == 0:
        return float('inf')

    min_pixels = float('inf')
    for car_pix in car_pixels[::5]:
        distances = np.sqrt(np.sum((obstacle_pixels - car_pix)**2, axis=1))
        min_pixels = min(min_pixels, np.min(distances))

    return min_pixels * 0.25


# ── helpers ──────────────────────────────────────────────────────────────────

def empty_grid(shape=(100, 100)):
    return np.zeros(shape, dtype=int)


def single_pixel_mask(shape, row, col):
    m = np.zeros(shape, dtype=bool)
    m[row, col] = True
    return m


def single_pixel_obs(shape, row, col):
    g = np.zeros(shape, dtype=int)
    g[row, col] = 1
    return g


# ── basic cases ──────────────────────────────────────────────────────────────

def test_collision_returns_zero():
    """Masks overlap → 0.0."""
    mask = single_pixel_mask((50, 50), 25, 25)
    obs = single_pixel_obs((50, 50), 25, 25)
    assert near_miss(mask, obs) == 0.0


def test_collision_partial_overlap_returns_zero():
    """Multiple pixels, at least one overlapping → still 0.0."""
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:15, 10:15] = True
    obs = np.zeros((50, 50), dtype=int)
    obs[14, 14] = 1      # corner overlap
    assert near_miss(mask, obs) == 0.0


def test_no_obstacles_returns_inf():
    """No obstacle pixels at all → inf."""
    mask = single_pixel_mask((50, 50), 25, 25)
    obs = empty_grid((50, 50))
    assert near_miss(mask, obs) == float('inf')


def test_empty_car_mask_no_pixels():
    """Empty car mask means no car pixels; obstacle exists but no car → inf.

    car_pixels = [] → the loop never runs → min_pixels stays inf.
    The function returns inf * 0.25... wait: actually it returns min_pixels * 0.25
    only if the loop runs. Let's check: if car_pixels is empty, the loop body
    never executes, min_pixels stays float('inf'), so we return inf * 0.25.
    float('inf') * 0.25 == float('inf'), so the test still passes.
    """
    mask = np.zeros((50, 50), dtype=bool)   # no car pixels
    obs = single_pixel_obs((50, 50), 25, 25)
    result = near_miss(mask, obs)
    assert result == float('inf')


# ── distance calculation ─────────────────────────────────────────────────────

def test_adjacent_pixel_distance_is_quarter_metre():
    """Car pixel at (25, 25), obstacle at (26, 25) — 1 pixel apart = 0.25 m."""
    mask = single_pixel_mask((50, 50), 25, 25)
    obs = single_pixel_obs((50, 50), 26, 25)
    assert abs(near_miss(mask, obs) - 0.25) < 1e-9


def test_four_pixel_separation_is_one_metre():
    """4 pixels apart (Euclidean, axis-aligned) = 1.0 m."""
    mask = single_pixel_mask((50, 50), 20, 20)
    obs = single_pixel_obs((50, 50), 24, 20)
    assert abs(near_miss(mask, obs) - 1.0) < 1e-9


def test_diagonal_distance():
    """3-4-5 right triangle: pixels (0,0) and (3,4) → dist=5 → 1.25 m."""
    mask = single_pixel_mask((50, 50), 0, 0)
    obs = single_pixel_obs((50, 50), 3, 4)
    assert abs(near_miss(mask, obs) - 1.25) < 1e-9


def test_multiple_obstacles_returns_minimum():
    """Returns distance to the NEAREST obstacle, not the farthest."""
    mask = single_pixel_mask((100, 100), 50, 50)
    obs = np.zeros((100, 100), dtype=int)
    obs[54, 50] = 1    # 4 pixels away → 1.0 m
    obs[90, 90] = 1    # far away
    result = near_miss(mask, obs)
    assert abs(result - 1.0) < 1e-9


def test_zero_distance_boundary_touching():
    """Mask pixel and obstacle pixel are not the same but share a face — still 1 pixel → 0.25 m."""
    mask = single_pixel_mask((50, 50), 10, 10)
    obs = single_pixel_obs((50, 50), 11, 10)
    assert abs(near_miss(mask, obs) - 0.25) < 1e-9


# ── subsampling awareness ────────────────────────────────────────────────────

def test_subsampling_does_not_undercount_when_nearest_pixel_sampled():
    """
    The [::5] subsampling can miss the true closest car pixel.
    This test ensures it at least finds the correct answer when the
    nearest car pixel IS in the sample (index 0 of the argwhere result).
    """
    mask = np.zeros((50, 50), dtype=bool)
    # Single car pixel at (20, 20) — will be index 0 in argwhere → always sampled
    mask[20, 20] = True
    obs = single_pixel_obs((50, 50), 22, 20)   # 2 pixels = 0.5 m
    assert abs(near_miss(mask, obs) - 0.5) < 1e-9


# ── grid border doesn't affect near_miss ─────────────────────────────────────

def test_obstacle_near_grid_edge():
    """Obstacle at grid boundary is found correctly."""
    mask = single_pixel_mask((50, 50), 25, 25)
    obs = single_pixel_obs((50, 50), 25, 45)   # 20 pixels = 5.0 m
    assert abs(near_miss(mask, obs) - 5.0) < 1e-9
