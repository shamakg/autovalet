"""
test_new_guidance.py — Unit tests for the three guidance improvements.

All tests run without GPU or the model checkpoint.
"""
import sys
import os
import math
import pytest
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
from v2 import TrajectoryPoint, Direction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_future_xy(ys, xs=None):
    """Build (1, T, 2) tensor where ys[t] is the lateral (y) coordinate."""
    T = len(ys)
    if xs is None:
        xs = list(range(1, T + 1))
    pts = torch.tensor([[xs, ys]], dtype=torch.float32).permute(0, 2, 1)  # (1, T, 2)
    return pts


def _lateral_penalties(future_xy_real, path_scale, std_x=20.0):
    """Replicate the two aisle-phase energy terms from diff_adapter._guidance_fn."""
    pos_pen = -(path_scale / std_x) * torch.mean(torch.abs(future_xy_real[:, :, 1]))
    dy = torch.diff(future_xy_real[:, :, 1], dim=1)
    hdg_pen = -(path_scale / std_x) * torch.mean(torch.abs(dy))
    return pos_pen, hdg_pen


# ---------------------------------------------------------------------------
# Test 1 – Heading penalty is negative for a trajectory with lateral velocity
# ---------------------------------------------------------------------------

def test_heading_penalty_negative_for_drifting_traj():
    """
    A trajectory that progressively drifts sideways (increasing y) must produce
    a strictly negative heading_penalty (energy punishes this behaviour).
    """
    # y increases by 0.5 m each step → car is rotating sideways
    ys = [i * 0.5 for i in range(20)]
    future_xy = _make_future_xy(ys)
    _, hdg_pen = _lateral_penalties(future_xy, path_scale=25000.0)
    assert hdg_pen.item() < 0.0, f"Expected negative heading penalty, got {hdg_pen.item()}"


def test_heading_penalty_near_zero_for_straight_traj():
    """
    A perfectly straight trajectory (constant y=0) must give heading_penalty ≈ 0
    because there is no lateral velocity at all.
    """
    ys = [0.0] * 20
    future_xy = _make_future_xy(ys)
    _, hdg_pen = _lateral_penalties(future_xy, path_scale=25000.0)
    assert abs(hdg_pen.item()) < 1e-4, f"Expected ~0 heading penalty, got {hdg_pen.item()}"


# ---------------------------------------------------------------------------
# Test 2 – Heading penalty catches turning even when position is centred
# ---------------------------------------------------------------------------

def test_heading_penalty_adds_extra_cost_for_turning_traj():
    """
    Two trajectories with the same final y-offset but different paths:
      A) Constant sideways drift (y increases every step)
      B) Position-only offset (constant y, no velocity)

    Trajectory A has lateral velocity throughout, so heading_penalty punishes
    it more than B.  The combined penalty for A must be strictly more negative
    than B's combined penalty even though they both end at the same lateral
    position.
    """
    # Trajectory A: constant drift, y goes 0→1 over 20 steps
    ys_drift = [i / 19.0 for i in range(20)]
    # Trajectory B: flat at constant y (same mean as drift, no velocity)
    mean_y = sum(ys_drift) / len(ys_drift)   # ≈ 0.5
    ys_flat = [mean_y] * 20

    xy_drift = _make_future_xy(ys_drift)
    xy_flat  = _make_future_xy(ys_flat)

    pos_drift, hdg_drift = _lateral_penalties(xy_drift, 25000.0)
    pos_flat,  hdg_flat  = _lateral_penalties(xy_flat,  25000.0)

    # pos_penalty is the same for both (same mean |y|)
    assert abs(pos_drift.item() - pos_flat.item()) < 1.0, (
        "pos penalties should be approximately equal for matched mean |y|"
    )

    # heading_penalty must be strictly worse for the drifting trajectory
    assert hdg_drift.item() < hdg_flat.item(), (
        "Drifting trajectory must have more negative heading_penalty than flat trajectory"
    )

    # Combined is strictly more negative for the drifting trajectory
    total_drift = pos_drift + hdg_drift
    total_flat  = pos_flat  + hdg_flat
    assert total_drift.item() < total_flat.item(), (
        "Combined penalty must be more negative for the drifting trajectory"
    )


# ---------------------------------------------------------------------------
# Test 3 – 10× scale produces 10× penalty magnitude
# ---------------------------------------------------------------------------

def test_scale_linearity():
    """
    Scaling PATH_SCALE by any factor must produce exactly that factor in the
    pos and heading penalties (both are linear in PATH_SCALE).
    """
    ys = [i * 0.3 for i in range(20)]
    future_xy = _make_future_xy(ys)

    pos_20, hdg_20   = _lateral_penalties(future_xy, path_scale=20.0)
    pos_200, hdg_200 = _lateral_penalties(future_xy, path_scale=200.0)

    ratio_pos = abs(pos_200.item()) / abs(pos_20.item())
    ratio_hdg = abs(hdg_200.item()) / abs(hdg_20.item())

    assert abs(ratio_pos - 10.0) < 0.01, f"pos_penalty ratio={ratio_pos:.3f}, expected 10.0"
    assert abs(ratio_hdg - 10.0) < 0.01, f"hdg_penalty ratio={ratio_hdg:.3f}, expected 10.0"


# ---------------------------------------------------------------------------
# Test 4 – Module-level constants match plan values
# ---------------------------------------------------------------------------

def test_guidance_scale_constants():
    assert da.GOAL_SCALE == 0.2,  f"GOAL_SCALE={da.GOAL_SCALE}, expected 0.2"
    assert da.PATH_SCALE == 8.0,  f"PATH_SCALE={da.PATH_SCALE}, expected 8.0"
    assert da.HALF_WIDTH == 1.0,  f"HALF_WIDTH={da.HALF_WIDTH}, expected 1.0"


# ---------------------------------------------------------------------------
# Test 5 – Pure pursuit selects first waypoint >= MAX_LOOKAHEAD_DIST
# ---------------------------------------------------------------------------

def test_pure_pursuit_selects_first_beyond_lookahead():
    """
    agent_interface.run_step_testbed picks the first TrajectoryPoint whose
    distance from ego >= MAX_LOOKAHEAD_DIST (12 m).  It must not skip forward
    past that point, and must not fall back to the last point when a closer
    candidate exists.

    We replicate the pure-pursuit selection loop directly (no CARLA needed).
    """
    MAX_LOOKAHEAD_DIST = 12.0

    # Fake ego location
    ego_x, ego_y = 0.0, 0.0

    # Build trajectory: points at x = 2, 4, 6, 8, 10, 12, 14 m
    # First point >= 12 m is at index 5 (x=12, dist=12)
    traj = [
        TrajectoryPoint(Direction.FORWARD, x=float(d), y=0.0, speed=1.5, angle=0.0)
        for d in [2, 4, 6, 8, 10, 12, 14]
    ]

    # Replicate the pure-pursuit logic from agent_interface.py
    import math
    lookahead = traj[-1]
    for pt in traj:
        if math.hypot(pt.x - ego_x, pt.y - ego_y) >= MAX_LOOKAHEAD_DIST:
            lookahead = pt
            break

    assert lookahead.x == 12.0, (
        f"Expected lookahead at x=12 (first >= {MAX_LOOKAHEAD_DIST}m), got x={lookahead.x}"
    )


def test_pure_pursuit_falls_back_to_last_when_all_close():
    """
    When every trajectory point is closer than MAX_LOOKAHEAD_DIST, the
    lookahead must fall back to traj[-1] (the farthest available).
    """
    MAX_LOOKAHEAD_DIST = 12.0
    ego_x, ego_y = 0.0, 0.0

    # All points within 8 m
    traj = [
        TrajectoryPoint(Direction.FORWARD, x=float(d), y=0.0, speed=1.5, angle=0.0)
        for d in [2, 4, 6, 8]
    ]

    import math
    lookahead = traj[-1]
    for pt in traj:
        if math.hypot(pt.x - ego_x, pt.y - ego_y) >= MAX_LOOKAHEAD_DIST:
            lookahead = pt
            break

    assert lookahead.x == 8.0, (
        f"Expected fallback to last point x=8.0, got x={lookahead.x}"
    )
