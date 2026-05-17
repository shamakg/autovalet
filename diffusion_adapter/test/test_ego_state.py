"""
test_ego_state.py — Isolate the ego velocity bug in adapter.py.

Tests that ego_current_state[4:6] (vx, vy) always equals (speed, 0)
in the ego-centric frame, regardless of global heading.

Run with:
    python test/test_ego_state.py

No CARLA, no model, no GPU needed.
"""

import sys, os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_AUTOVALET_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _AUTOVALET_ROOT)

from utils.coord_utils import carla_transform_to_standard, carla_velocity_to_standard


def build_ego_current_state_buggy(speed, heading_deg):
    """Replicates the CURRENT (buggy) logic from adapter.py:246."""
    _, _, heading_rad = carla_transform_to_standard(0, 0, heading_deg)
    # cur.angle is stored in radians; adapter.py does cos/sin directly on it
    std_vx, std_vy = carla_velocity_to_standard(
        speed * np.cos(heading_rad),
        speed * np.sin(heading_rad),
    )
    return np.array([0., 0., 1., 0., std_vx, std_vy, 0., 0., 0., 0.], dtype=np.float32)


def build_ego_current_state_fixed(speed, heading_deg):
    """The CORRECT ego-centric velocity: always (speed, 0) in ego frame."""
    return np.array([0., 0., 1., 0., speed, 0., 0., 0., 0., 0.], dtype=np.float32)


def check(cond, tag, detail=''):
    status = '[PASS]' if cond else '[FAIL]'
    print(f"  {status} {tag}  {detail}")
    return cond


def test_ego_velocity_is_forward():
    """
    In ego-centric frame the car always points forward (heading=0),
    so vx must equal speed and vy must equal 0, for any global heading.
    """
    print("\n=== Ego velocity correctness (heading-parametrized) ===")
    print("Checks that vx=speed, vy=0 in ego frame for any global heading.\n")

    SPEED = 5.0
    headings = [0.0, 45.0, 90.0, 135.0, 180.0, 270.0]
    all_ok = True

    for hdeg in headings:
        buggy = build_ego_current_state_buggy(SPEED, hdeg)
        fixed = build_ego_current_state_fixed(SPEED, hdeg)

        vx_bug, vy_bug = buggy[4], buggy[5]
        vx_fix, vy_fix = fixed[4], fixed[5]

        # Expected: (SPEED, 0)
        ok_vx = np.isclose(vx_bug, SPEED, atol=1e-4)
        ok_vy = np.isclose(vy_bug, 0.0,   atol=1e-4)

        # Show both values so the bug is obvious
        print(f"  heading={hdeg:6.1f}°  buggy vx={vx_bug:+.4f} vy={vy_bug:+.4f}"
              f"  |  fixed vx={vx_fix:+.4f} vy={vy_fix:+.4f}")

        r1 = check(ok_vx, f"heading={hdeg}° → vx == speed ({SPEED})",
                   f"got vx={vx_bug:.4f}")
        r2 = check(ok_vy, f"heading={hdeg}° → vy == 0",
                   f"got vy={vy_bug:.4f}")
        all_ok &= (r1 and r2)

    return all_ok


def test_heading_90_is_worst_case():
    """
    At 90° the bug is maximally wrong: vx≈0 and vy≈speed instead of (speed, 0).
    This is the canonical CARLA spawn heading and explains the perpendicular path.
    """
    print("\n=== 90° heading — worst-case bug demonstration ===")
    SPEED = 5.0
    buggy = build_ego_current_state_buggy(SPEED, 90.0)
    print(f"  Buggy  ego_current_state[4:6] = ({buggy[4]:.4f}, {buggy[5]:.4f})")
    print(f"  Expect ego_current_state[4:6] = ({SPEED:.4f}, 0.0000)")

    ok_vx = check(np.isclose(buggy[4], SPEED, atol=1e-4),
                  "vx == speed at 90°", f"got {buggy[4]:.4f}")
    ok_vy = check(np.isclose(buggy[5], 0.0,   atol=1e-4),
                  "vy == 0    at 90°", f"got {buggy[5]:.4f}")

    if not (ok_vx and ok_vy):
        print("\n  *** BUG CONFIRMED: velocity is in global frame, not ego frame.")
        print("      At 90° the model receives (vx≈0, vy=speed) — car appears to")
        print("      move sideways, which causes the circular spinning path. ***")

    return ok_vx and ok_vy


if __name__ == '__main__':
    r1 = test_heading_90_is_worst_case()
    r2 = test_ego_velocity_is_forward()

    print(f"\n{'='*55}")
    passed = sum([r1, r2])
    print(f"Results: {passed}/2 test groups passed")
    if passed < 2:
        print("BUG ISOLATED — fix adapter.py:246-255 (see plan).")
    else:
        print("All checks pass — bug is fixed.")
