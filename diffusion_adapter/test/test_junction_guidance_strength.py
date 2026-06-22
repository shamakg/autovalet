"""
test_junction_guidance_strength.py
------------------------------------
Diagnoses why the car still turns South at the junction even after the bearing
fix (which places the destination exactly straight ahead).

OBSERVED FAILURE (from CARLA run):
  After trigger (bearing=6.6°, dest_ego=(7.65,0)):
  pred[-1] ego=(42.79, -13.24)  — model predicts 13m South of destination row
  lookahead CARLA=(297.05,-235.58) — 2m south of destination row y=-232.73
  → pure pursuit steers into parked vehicles, collision in ~50 ticks

ROOT CAUSE QUESTION:
  Is the guidance gradient large enough to shift the trajectory from
  (42.79, -13.24) toward (7.65, 0)?

  At waypoint t=20 (approx 12m forward, 3.3m south), only path_energy acts.
  We measure its northward gradient and compare to the displacement needed:
    required shift in y_norm: 3.31/20 = 0.166 units

  If the per-step gradient is too small, we need to scale up PATH_SCALE
  (or return a multiplied energy from guidance_fn to amplify all gradients).

Tests:
  1. Gradient direction at a South-drifted trajectory: must be northward (+y)
  2. Gradient magnitude with current PATH_SCALE=2500: report baseline
  3. Gradient magnitude with PATH_SCALE=25000 (10×): report amplified value
  4. Final-waypoint goal_energy gradient: measure northward component
  5. Combined gradient with 10× scales is N× larger than needed displacement

No model, no CARLA server required.
"""

import sys, os
import numpy as np
import torch
import pytest

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MEAN_XY     = np.array([10.0, 0.0])
STD_XY      = np.array([20.0, 20.0])

# Current scales (before fix)
GOAL_SCALE_OLD  = 5000.0
PATH_SCALE_OLD  = 2500.0

# Proposed scales (10× amplification)
GOAL_SCALE_NEW  = 50000.0
PATH_SCALE_NEW  = 25000.0

T_STEPS     = 80    # DiffusionPlanner prediction horizon

# Scenario values at trigger (from CARLA run log)
DEST_EGO    = np.array([7.65, 0.0])     # destination in bearing ego frame
# Model trajectory at trigger: 80 waypoints from ego(0,0) to ego(42.79,-13.24)
# Approximated linearly for testing purposes
TRAJ_END    = np.array([42.79, -13.24])


def _make_traj_real(traj_end=TRAJ_END, n=T_STEPS):
    """Linear trajectory from ego(0,0) to traj_end in real metres."""
    xs = np.linspace(0.0, traj_end[0], n)
    ys = np.linspace(0.0, traj_end[1], n)
    return np.stack([xs, ys], axis=1)   # (T, 2)


def _path_energy_grad_at_t(t_idx, traj_real, route_y=0.0, path_scale=PATH_SCALE_OLD):
    """
    Compute path_energy gradient at waypoint t_idx w.r.t. the normalised
    trajectory coordinate.

    path_energy = -(PATH_SCALE / STD_X) * mean_{t}(min_dist_to_route)
    We approximate the route as a horizontal line at y=route_y
    (the destination row in bearing ego frame).
    """
    sxy = torch.tensor(STD_XY, dtype=torch.float32)
    mxy = torch.tensor(MEAN_XY, dtype=torch.float32)

    # Build normalised trajectory tensor (T, 2), only t_idx has gradient
    traj_norm = (traj_real - MEAN_XY) / STD_XY
    fn = torch.tensor(traj_norm[t_idx], dtype=torch.float32, requires_grad=True)
    fn_real = fn * sxy + mxy   # (2,) real metres at waypoint t_idx

    # Approximate: route is the segment from ego-origin to traj_real[t_idx] at y=route_y
    # i.e., nearest route point at this x is (fn_real[0], route_y)
    nearest_route_y = torch.tensor(route_y, dtype=torch.float32)
    dist_to_route   = torch.abs(fn_real[1] - nearest_route_y)   # scalar
    path_energy     = -(path_scale / STD_XY[0]) * dist_to_route / T_STEPS
    path_energy.backward()
    return path_energy.item(), fn.grad.detach().numpy().copy()


def _goal_energy_grad_at_final(traj_real, dest_ego=DEST_EGO, goal_scale=GOAL_SCALE_OLD):
    """Gradient of goal_energy at the final trajectory waypoint."""
    sxy = torch.tensor(STD_XY, dtype=torch.float32)
    mxy = torch.tensor(MEAN_XY, dtype=torch.float32)
    dst = torch.tensor(dest_ego, dtype=torch.float32)

    final_norm_np = (traj_real[-1] - MEAN_XY) / STD_XY
    fn = torch.tensor(final_norm_np, dtype=torch.float32, requires_grad=True)
    fn_real     = fn * sxy + mxy
    goal_dist   = torch.norm(fn_real - dst)
    goal_energy = -(goal_scale / STD_XY[0]) * goal_dist
    goal_energy.backward()
    return goal_energy.item(), fn.grad.detach().numpy().copy()


# ===========================================================================
# Test 1: path_energy gradient direction at South-drifted waypoint is Northward
# ===========================================================================

def test_path_energy_direction_is_northward():
    """
    At waypoint t=20 the trajectory is 3.3m South of the destination row.
    path_energy gradient must point NORTHWARD (+y in bearing ego frame).
    """
    traj_real = _make_traj_real()
    t_idx = 20   # ≈ 10.7m forward, 3.3m south

    _, grad = _path_energy_grad_at_t(t_idx, traj_real)
    print(f"\n[path_energy t=20] ego=({traj_real[t_idx,0]:.2f},{traj_real[t_idx,1]:.2f})")
    print(f"  grad_x={grad[0]:.4f}, grad_y={grad[1]:.4f}")

    assert grad[1] > 0, f"Expected northward (+y) gradient, got grad_y={grad[1]:.4f}"
    assert abs(grad[0]) < abs(grad[1]) * 0.1, \
        f"Expected gradient mostly in y, got grad=({grad[0]:.4f},{grad[1]:.4f})"


# ===========================================================================
# Test 2: path_energy gradient magnitude baseline (PATH_SCALE=2500)
# ===========================================================================

def test_path_energy_magnitude_baseline():
    """
    Report the baseline path_energy gradient magnitude at t=20.
    The normalized displacement needed at t=20 is ΔyNorm = 3.31/20 = 0.166.
    We check whether the gradient magnitude is in the right ballpark
    (not necessarily > displacement — DPM integration matters).
    """
    traj_real = _make_traj_real()
    t_idx = 20
    displacement_needed_norm = abs(traj_real[t_idx, 1]) / STD_XY[1]

    _, grad = _path_energy_grad_at_t(t_idx, traj_real, path_scale=PATH_SCALE_OLD)
    print(f"\n[baseline PATH_SCALE={PATH_SCALE_OLD}]")
    print(f"  displacement_needed_norm={displacement_needed_norm:.4f}")
    print(f"  grad_y (normalised)={grad[1]:.4f}")
    print(f"  ratio grad_y/displacement_needed={grad[1]/max(abs(displacement_needed_norm), 1e-8):.1f}")

    # Just verify gradient is positive (northward) and finite
    assert grad[1] > 0
    assert np.isfinite(grad[1])


# ===========================================================================
# Test 3: 10× scales produce 10× gradient
# ===========================================================================

def test_10x_scales_give_10x_gradient():
    """
    GOAL_SCALE=50000, PATH_SCALE=25000 must produce exactly 10× the gradient
    magnitude of the original scales.  This verifies the implementation is
    trivially correct (autograd scales linearly with energy).
    """
    traj_real = _make_traj_real()

    _, path_grad_old = _path_energy_grad_at_t(20, traj_real, path_scale=PATH_SCALE_OLD)
    _, path_grad_new = _path_energy_grad_at_t(20, traj_real, path_scale=PATH_SCALE_NEW)
    _, goal_grad_old = _goal_energy_grad_at_final(traj_real, goal_scale=GOAL_SCALE_OLD)
    _, goal_grad_new = _goal_energy_grad_at_final(traj_real, goal_scale=GOAL_SCALE_NEW)

    path_ratio = abs(path_grad_new[1]) / max(abs(path_grad_old[1]), 1e-8)
    goal_ratio = abs(goal_grad_new[1]) / max(abs(goal_grad_old[1]), 1e-8)

    print(f"\n[scale ratio test]")
    print(f"  path_grad_y: {path_grad_old[1]:.4f} → {path_grad_new[1]:.4f}  ratio={path_ratio:.2f}")
    print(f"  goal_grad_y: {goal_grad_old[1]:.4f} → {goal_grad_new[1]:.4f}  ratio={goal_ratio:.2f}")

    assert abs(path_ratio - 10.0) < 0.01, f"path scale ratio should be 10.0, got {path_ratio:.3f}"
    assert abs(goal_ratio - 10.0) < 0.01, f"goal scale ratio should be 10.0, got {goal_ratio:.3f}"


# ===========================================================================
# Test 4: goal_energy at final waypoint is northward (+y) in bearing frame
# ===========================================================================

def test_goal_energy_at_final_is_northward():
    """
    With traj_end=(42.79,-13.24) and dest=(7.65,0), the final waypoint is
    SOUTH and EAST of the destination.  goal_energy gradient must be:
      - northward (+y): pulling South-displaced final waypoint toward dest row
      - backward (-x): pulling far-ahead final waypoint back toward dest x
    """
    traj_real = _make_traj_real()
    _, grad = _goal_energy_grad_at_final(traj_real, goal_scale=GOAL_SCALE_OLD)

    print(f"\n[goal_energy at final] traj_end={TRAJ_END}, dest_ego={DEST_EGO}")
    print(f"  grad_x={grad[0]:.1f}, grad_y={grad[1]:.1f}")

    # Final is 13.24m South of dest → gradient should pull North (+y)
    assert grad[1] > 0, f"Expected northward gradient at South-drifted final, got grad_y={grad[1]:.1f}"
    # Final is 35m East of dest → gradient should pull back West (-x)
    assert grad[0] < 0, f"Expected backward (-x) gradient, got grad_x={grad[0]:.1f}"


# ===========================================================================
# Test 5: With 10× scales, the NEW guidance gradient exceeds the required
#         displacement by a factor of ≥ 5 (enough margin to overcome prior)
# ===========================================================================

def test_amplified_path_gradient_sufficient_for_junction():
    """
    The path_energy gradient at t=20 with PATH_SCALE=25000 should be
    significantly larger than the normalised displacement needed (0.166).
    Factor-of-5 margin means guidance SHOULD overcome the model prior.

    Note: this doesn't guarantee it will (DPM integration is non-trivial),
    but it verifies the gradient is in the right order of magnitude.
    """
    traj_real = _make_traj_real()
    t_idx = 20
    displacement_needed_norm = abs(traj_real[t_idx, 1]) / STD_XY[1]  # 0.166

    _, grad_new = _path_energy_grad_at_t(t_idx, traj_real, path_scale=PATH_SCALE_NEW)
    grad_y_new = grad_new[1]

    ratio = grad_y_new / max(displacement_needed_norm, 1e-8)
    print(f"\n[amplified guidance]")
    print(f"  displacement_needed_norm={displacement_needed_norm:.4f}")
    print(f"  path grad_y with PATH_SCALE={PATH_SCALE_NEW}: {grad_y_new:.4f}")
    print(f"  ratio = {ratio:.1f}× (need ≥5 for sufficient margin)")

    assert ratio >= 5.0, \
        f"Amplified path gradient {grad_y_new:.4f} is only {ratio:.1f}× " \
        f"the needed displacement {displacement_needed_norm:.4f} — may be insufficient"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
