"""
test_agent_pipeline.py — Tests for build_neighbor_agents_past.

No CARLA, no model, no GPU needed.
Requires nuplan-devkit to be on the path (same as test_model_smoke.py).

Run with:
    /home/sumesh/envs/simlingo/bin/python test/test_agent_pipeline.py
"""

import sys, os
import numpy as np

_HERE           = os.path.dirname(os.path.abspath(__file__))
_ADAPTER        = os.path.dirname(_HERE)
_AUTOVALET_ROOT = os.path.dirname(_ADAPTER)
_NUPLAN_ROOT    = os.path.join(_ADAPTER, "nuplan-devkit")
_DP_ROOT        = os.path.join(_ADAPTER, "Diffusion-Planner")

for _p in [_ADAPTER, _AUTOVALET_ROOT, _NUPLAN_ROOT, _DP_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.agent_process import (
    AgentState, AgentHistoryBuffer, build_neighbor_agents_past,
    NUM_AGENTS, NUM_PAST_STEPS,
)
from utils.coord_utils import carla_transform_to_standard, carla_velocity_to_standard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check(cond, tag, detail=''):
    status = '[PASS]' if cond else '[FAIL]'
    print(f"  {status} {tag}  {detail}")
    return cond


def _anchor(ego_x=0.0, ego_y=0.0, ego_h=0.0):
    """Ego pose as [x, y, heading] in standard frame."""
    return np.array([ego_x, ego_y, ego_h], dtype=np.float64)


def _make_state(actor_id, x, y, heading=0.0, vx=0.0, vy=0.0,
                width=2.0, length=4.5, agent_type='vehicle'):
    return AgentState(actor_id, x, y, heading, vx, vy, width, length, agent_type)


# ---------------------------------------------------------------------------
# Test 1 — empty history → all zeros
# ---------------------------------------------------------------------------

def test_empty_history():
    print("\n=== Test 1: Empty history → zeros ===")
    buf = AgentHistoryBuffer()
    out = build_neighbor_agents_past(buf, _anchor())

    ok_shape = check(out.shape == (NUM_AGENTS, NUM_PAST_STEPS, 11),
                     "output shape == (32, 21, 11)", f"got {out.shape}")
    ok_zeros = check(np.all(out == 0),
                     "all entries are zero for empty buffer")
    return ok_shape and ok_zeros


# ---------------------------------------------------------------------------
# Test 2 — single vehicle 10 m ahead in ego frame
# ---------------------------------------------------------------------------

def test_single_vehicle_ahead():
    print("\n=== Test 2: Single vehicle 10 m ahead ===")
    EGO_X, EGO_Y, EGO_H = 100.0, 50.0, 0.0   # ego faces +x
    AGENT_X = EGO_X + 10.0                     # 10 m ahead
    AGENT_Y = EGO_Y

    buf = AgentHistoryBuffer()
    state = _make_state(1, AGENT_X, AGENT_Y, heading=EGO_H,
                        vx=1.0, vy=0.0, width=2.0, length=4.5)
    for _ in range(NUM_PAST_STEPS):
        buf.update([state])

    anchor = _anchor(EGO_X, EGO_Y, EGO_H)
    out    = build_neighbor_agents_past(buf, anchor)  # (32, 21, 11)

    # Find the non-zero agent row
    row = None
    for i in range(NUM_AGENTS):
        if not np.all(out[i] == 0):
            row = out[i]
            break

    if row is None:
        print("  [FAIL] No non-zero agent row found")
        return False

    # Last timestep: [x, y, cos_h, sin_h, vx, vy, width, length, is_veh, is_ped, is_bike]
    last = row[-1]
    ok_x   = check(abs(last[0] - 10.0) < 0.5, "ego-centric x ≈ 10.0", f"got {last[0]:.3f}")
    ok_y   = check(abs(last[1]) < 0.5,          "ego-centric y ≈ 0.0",  f"got {last[1]:.3f}")
    ok_veh = check(last[8] == 1.0,              "is_vehicle == 1",      f"got {last[8]}")
    ok_ped = check(last[9] == 0.0,              "is_pedestrian == 0",   f"got {last[9]}")
    ok_bik = check(last[10] == 0.0,             "is_bicycle == 0",      f"got {last[10]}")
    return ok_x and ok_y and ok_veh and ok_ped and ok_bik


# ---------------------------------------------------------------------------
# Test 3 — agent type encoding (vehicle, pedestrian, bicycle)
# ---------------------------------------------------------------------------

def test_agent_type_encoding():
    print("\n=== Test 3: Agent type one-hot encoding ===")
    EGO_X, EGO_Y = 0.0, 0.0

    # Place agents at distinct offsets so they don't collide
    states = [
        _make_state(1, EGO_X + 10.0, EGO_Y + 0.0,  agent_type='vehicle'),
        _make_state(2, EGO_X + 10.0, EGO_Y + 5.0,  agent_type='pedestrian'),
        _make_state(3, EGO_X + 10.0, EGO_Y + 10.0, agent_type='bicycle'),
    ]

    buf = AgentHistoryBuffer()
    for _ in range(NUM_PAST_STEPS):
        buf.update(states)

    out = build_neighbor_agents_past(buf, _anchor())

    # Collect non-zero agent rows and check their type bits
    type_bits = []
    for i in range(NUM_AGENTS):
        if not np.all(out[i] == 0):
            type_bits.append(tuple(out[i, -1, 8:11].tolist()))

    # Each type should appear exactly once
    found_veh = (1.0, 0.0, 0.0) in type_bits
    found_ped = (0.0, 1.0, 0.0) in type_bits
    found_bik = (0.0, 0.0, 1.0) in type_bits

    ok_v = check(found_veh, "vehicle type (1,0,0) present", str(type_bits))
    ok_p = check(found_ped, "pedestrian type (0,1,0) present", str(type_bits))
    ok_b = check(found_bik, "bicycle type (0,0,1) present", str(type_bits))
    return ok_v and ok_p and ok_b


# ---------------------------------------------------------------------------
# Test 4 — history padding (only 1 frame → repeated to 21)
# ---------------------------------------------------------------------------

def test_history_padding():
    print("\n=== Test 4: History padding (1 frame → 21 steps) ===")
    buf = AgentHistoryBuffer()
    # Add exactly 1 frame
    buf.update([_make_state(1, 10.0, 0.0, vx=2.0)])

    out = build_neighbor_agents_past(buf, _anchor())

    ok_shape = check(out.shape == (NUM_AGENTS, NUM_PAST_STEPS, 11),
                     "output shape still (32, 21, 11)", f"got {out.shape}")

    row = None
    for i in range(NUM_AGENTS):
        if not np.all(out[i] == 0):
            row = out[i]
            break

    if row is None:
        print("  [FAIL] No non-zero agent row found — padding did not populate")
        return False

    # First and last timestep should both be non-zero (oldest state repeated)
    ok_t0 = check(not np.all(row[0] == 0),  "timestep 0 is non-zero (padded with oldest)")
    ok_t1 = check(not np.all(row[-1] == 0), "timestep -1 (current) is non-zero")
    return ok_shape and ok_t0 and ok_t1


# ---------------------------------------------------------------------------
# Test 5 — agent to the LEFT and RIGHT of ego (y-axis sign check)
# ---------------------------------------------------------------------------

def test_lateral_placement():
    """
    Ego at (0,0) facing +X (heading=0).
    Left agent at (0, +5)  → ego-centric y should be ≈ +5
    Right agent at (0, -5) → ego-centric y should be ≈ -5

    This verifies the coordinate convention used in agent_process.py is
    right-handed (nuPlan): +Y = left of ego when facing +X.
    """
    print("\n=== Test 5: Lateral placement (left/right y-sign) ===")
    EGO_X, EGO_Y, EGO_H = 0.0, 0.0, 0.0

    left_state  = _make_state(10, EGO_X,        EGO_Y + 5.0, heading=EGO_H)
    right_state = _make_state(11, EGO_X,        EGO_Y - 5.0, heading=EGO_H)

    buf = AgentHistoryBuffer()
    for _ in range(NUM_PAST_STEPS):
        buf.update([left_state, right_state])

    anchor = _anchor(EGO_X, EGO_Y, EGO_H)
    out    = build_neighbor_agents_past(buf, anchor)

    ys = []
    for i in range(NUM_AGENTS):
        if not np.all(out[i] == 0):
            ys.append(out[i, -1, 1])  # ego-centric y at last timestep

    # We expect one agent with y≈+5 and one with y≈-5
    has_left  = any(abs(y - 5.0) < 0.5 for y in ys)
    has_right = any(abs(y + 5.0) < 0.5 for y in ys)

    ok_l = check(has_left,  "left agent  → ego-centric y ≈ +5", f"ys={[f'{y:.2f}' for y in ys]}")
    ok_r = check(has_right, "right agent → ego-centric y ≈ -5", f"ys={[f'{y:.2f}' for y in ys]}")
    return ok_l and ok_r


# ---------------------------------------------------------------------------
# Test 6 — rotated ego heading (π/2: ego faces +Y)
# ---------------------------------------------------------------------------

def test_rotated_ego_heading():
    """
    Ego at (0,0) heading=π/2 (facing +Y).
    Agent at global (0, +10) — directly AHEAD of ego.
    Expected ego-centric: x ≈ +10, y ≈ 0.
    """
    print("\n=== Test 6: Rotated ego heading (π/2) ===")
    EGO_H = np.pi / 2.0

    # agent is 10 m ahead: ego faces +Y, so ahead = (0, +10)
    state = _make_state(20, 0.0, 10.0, heading=EGO_H)
    buf   = AgentHistoryBuffer()
    for _ in range(NUM_PAST_STEPS):
        buf.update([state])

    anchor = _anchor(0.0, 0.0, EGO_H)
    out    = build_neighbor_agents_past(buf, anchor)

    row = None
    for i in range(NUM_AGENTS):
        if not np.all(out[i] == 0):
            row = out[i]
            break

    if row is None:
        print("  [FAIL] No non-zero agent row found")
        return False

    last = row[-1]
    ok_x = check(abs(last[0] - 10.0) < 0.5, "ego-centric x ≈ +10 (agent ahead)", f"got {last[0]:.3f}")
    ok_y = check(abs(last[1])        < 0.5, "ego-centric y ≈  0  (agent ahead)", f"got {last[1]:.3f}")
    return ok_x and ok_y


# ---------------------------------------------------------------------------
# Test 7 — lateral velocity transform
# ---------------------------------------------------------------------------

def test_lateral_velocity():
    """
    Agent with global velocity (vx=0, vy=+1) and ego heading=0.
    Ego-centric velocity should be (vx=0, vy=+1) — pure lateral.

    Agent with global velocity (vx=0, vy=+1) and ego heading=π/2 (ego faces +Y).
    Ego-centric velocity: forward component = vx_g*cos(h)+vy_g*sin(h) = 0+1 = +1
                          lateral component = vy_g*cos(h)-vx_g*sin(h) = 0-0  =  0
    So ego-centric (vx≈+1, vy≈0).
    """
    print("\n=== Test 7: Lateral velocity transform ===")
    all_ok = True

    # --- sub-case A: ego heading=0, agent moving in +Y ---
    state_a = _make_state(30, 10.0, 0.0, heading=0.0, vx=0.0, vy=1.0)
    buf_a   = AgentHistoryBuffer()
    for _ in range(NUM_PAST_STEPS):
        buf_a.update([state_a])
    out_a = build_neighbor_agents_past(buf_a, _anchor(0.0, 0.0, 0.0))
    row_a = next((out_a[i] for i in range(NUM_AGENTS) if not np.all(out_a[i] == 0)), None)
    if row_a is None:
        print("  [FAIL] sub-case A: no non-zero row")
        return False
    la = row_a[-1]
    ok_a_vx = check(abs(la[4]) < 0.1,       "h=0, vy_global=+1 → ego vx ≈ 0",  f"got {la[4]:.3f}")
    ok_a_vy = check(abs(la[5] - 1.0) < 0.1, "h=0, vy_global=+1 → ego vy ≈ +1", f"got {la[5]:.3f}")
    all_ok  = all_ok and ok_a_vx and ok_a_vy

    # --- sub-case B: ego heading=π/2, agent moving in +Y (= forward) ---
    EGO_H   = np.pi / 2.0
    state_b = _make_state(31, 0.0, 10.0, heading=EGO_H, vx=0.0, vy=1.0)
    buf_b   = AgentHistoryBuffer()
    for _ in range(NUM_PAST_STEPS):
        buf_b.update([state_b])
    out_b = build_neighbor_agents_past(buf_b, _anchor(0.0, 0.0, EGO_H))
    row_b = next((out_b[i] for i in range(NUM_AGENTS) if not np.all(out_b[i] == 0)), None)
    if row_b is None:
        print("  [FAIL] sub-case B: no non-zero row")
        return False
    lb = row_b[-1]
    ok_b_vx = check(abs(lb[4] - 1.0) < 0.1, "h=π/2, vy_global=+1 → ego vx ≈ +1 (forward)", f"got {lb[4]:.3f}")
    ok_b_vy = check(abs(lb[5])        < 0.1, "h=π/2, vy_global=+1 → ego vy ≈  0 (no lateral)", f"got {lb[5]:.3f}")
    all_ok  = all_ok and ok_b_vx and ok_b_vy

    return all_ok


# ---------------------------------------------------------------------------
# Test 8 — coord_utils passthrough contract
# ---------------------------------------------------------------------------

def test_carla_passthrough_conversion():
    """
    The whole diffusion pipeline uses CARLA coordinates as-is (passthrough).
    This test locks in that contract so any future change to coord_utils is
    caught immediately.

    carla_transform_to_standard: x→x, y→y, heading_deg→rad (no sign change)
    carla_velocity_to_standard:  vx→vx, vy→vy
    """
    print("\n=== Test 8: coord_utils passthrough contract ===")

    sx, sy, sh = carla_transform_to_standard(10.0, -5.0, 90.0)
    ok_x  = check(abs(sx - 10.0) < 1e-9,       "x preserved",               f"got {sx}")
    ok_y  = check(abs(sy - (-5.0)) < 1e-9,      "y preserved (no flip)",     f"got {sy}")
    ok_h  = check(abs(sh - np.pi/2) < 1e-9,     "heading: 90° → π/2 rad",   f"got {sh:.6f}")

    svx, svy = carla_velocity_to_standard(3.0, -1.0)
    ok_vx = check(abs(svx - 3.0) < 1e-9,        "vx preserved",              f"got {svx}")
    ok_vy = check(abs(svy - (-1.0)) < 1e-9,     "vy preserved (no flip)",    f"got {svy}")

    return ok_x and ok_y and ok_h and ok_vx and ok_vy


# ---------------------------------------------------------------------------
# Test 9 — CARLA parked car relative position (parking lot scenario)
# ---------------------------------------------------------------------------

def test_carla_actor_relative_position():
    """
    Replicates the actual Town04 parking scenario coordinates:
      CARLA ego  : (285.6, -243.7, yaw=90°)  — ego faces +Y (CARLA convention)
      CARLA car A: (285.6, -183.7)            — 60m directly ahead
      CARLA car B: (290.6, -183.7)            — 60m ahead + 5m to global +X

    After passthrough conversion and ego-centric transform:
      Car A: ego-x ≈ +60, ego-y ≈  0   (pure forward)
      Car B: ego-x ≈ +60, ego-y ≈ -5   (ahead and lateral)

    The lateral sign (ego-y = -5) is geometrically consistent with the
    passthrough convention used everywhere else in the pipeline.
    """
    print("\n=== Test 9: CARLA parked car relative position (Town04 scenario) ===")

    # Build anchor from CARLA ego pose
    ego_cx, ego_cy, ego_yaw = 285.6, -243.7, 90.0
    ax, ay, ah = carla_transform_to_standard(ego_cx, ego_cy, ego_yaw)
    anchor = _anchor(ax, ay, ah)

    def _make_carla_state(actor_id, cx, cy, cyaw_deg=0.0):
        sx, sy, sh = carla_transform_to_standard(cx, cy, cyaw_deg)
        return _make_state(actor_id, sx, sy, heading=sh)

    car_a = _make_carla_state(100, 285.6, -183.7)   # 60m ahead in +Y CARLA
    car_b = _make_carla_state(101, 290.6, -183.7)   # 60m ahead + 5m in +X CARLA

    buf = AgentHistoryBuffer()
    for _ in range(NUM_PAST_STEPS):
        buf.update([car_a, car_b])

    out = build_neighbor_agents_past(buf, anchor)

    rows = {i: out[i] for i in range(NUM_AGENTS) if not np.all(out[i] == 0)}
    if len(rows) < 2:
        print(f"  [FAIL] Expected 2 non-zero rows, got {len(rows)}")
        return False

    # Collect ego-centric (x, y) for both cars
    pts = [row[-1, :2] for row in rows.values()]

    # Car A: (≈60, ≈0)
    a = min(pts, key=lambda p: abs(p[1]))   # smallest |y| = car A
    ok_a_x = check(abs(a[0] - 60.0) < 1.0, "car A ego-x ≈ +60 (ahead)",  f"got {a[0]:.2f}")
    ok_a_y = check(abs(a[1])        < 1.0, "car A ego-y ≈  0  (no drift)", f"got {a[1]:.2f}")

    # Car B: (≈60, ≈-5)
    b = max(pts, key=lambda p: abs(p[1]))   # largest |y| = car B
    ok_b_x = check(abs(b[0] - 60.0) < 1.0, "car B ego-x ≈ +60 (ahead)",   f"got {b[0]:.2f}")
    ok_b_y = check(abs(b[1] + 5.0)  < 1.0, "car B ego-y ≈ -5  (lateral)", f"got {b[1]:.2f}")

    return ok_a_x and ok_a_y and ok_b_x and ok_b_y


# ---------------------------------------------------------------------------
# Test 10 — CARLA pedestrian velocity in ego frame
# ---------------------------------------------------------------------------

def test_carla_pedestrian_velocity():
    """
    CARLA pedestrian moving in +Y direction (same direction as ego at yaw=90°).
    After passthrough: std_vx=0, std_vy=+1.
    After ego-centric velocity rotation (heading=π/2):
        ego_vx = vx*cos(π/2) + vy*sin(π/2) = 0 + 1 = +1  (forward)
        ego_vy = vy*cos(π/2) - vx*sin(π/2) = 0 - 0  =  0  (no lateral)

    A pedestrian crossing perpendicular (CARLA vx=+1, vy=0):
        std_vx=1, std_vy=0
        ego_vx = 1*cos(π/2) + 0*sin(π/2) = 0      (no forward component)
        ego_vy = 0*cos(π/2) - 1*sin(π/2) = -1      (lateral, moving right in ego frame)
    """
    print("\n=== Test 10: CARLA pedestrian velocity in ego frame ===")

    EGO_YAW = 90.0  # ego faces +Y CARLA
    ax, ay, ah = carla_transform_to_standard(0.0, 0.0, EGO_YAW)
    anchor = _anchor(ax, ay, ah)

    all_ok = True

    # --- sub-case A: pedestrian moving forward (+Y CARLA = same as ego) ---
    svx_a, svy_a = carla_velocity_to_standard(0.0, 1.0)
    std_x_a, std_y_a, std_h_a = carla_transform_to_standard(0.0, 10.0, EGO_YAW)
    state_a = _make_state(200, std_x_a, std_y_a, heading=std_h_a,
                          vx=svx_a, vy=svy_a, agent_type='pedestrian')
    buf_a = AgentHistoryBuffer()
    for _ in range(NUM_PAST_STEPS):
        buf_a.update([state_a])
    out_a = build_neighbor_agents_past(buf_a, anchor)
    row_a = next((out_a[i] for i in range(NUM_AGENTS) if not np.all(out_a[i] == 0)), None)
    if row_a is None:
        print("  [FAIL] sub-case A: no non-zero row"); return False
    la = row_a[-1]
    ok_a = check(abs(la[4] - 1.0) < 0.1 and abs(la[5]) < 0.1,
                 "ped moving forward (+Y CARLA) → ego vx≈+1, vy≈0",
                 f"got vx={la[4]:.3f} vy={la[5]:.3f}")
    ok_ped_a = check(la[9] == 1.0, "agent type = pedestrian", f"got {la[9]}")
    all_ok = all_ok and ok_a and ok_ped_a

    # --- sub-case B: pedestrian crossing (+X CARLA = perpendicular to ego) ---
    svx_b, svy_b = carla_velocity_to_standard(1.0, 0.0)
    std_x_b, std_y_b, std_h_b = carla_transform_to_standard(0.0, 10.0, EGO_YAW)
    state_b = _make_state(201, std_x_b, std_y_b, heading=std_h_b,
                          vx=svx_b, vy=svy_b, agent_type='pedestrian')
    buf_b = AgentHistoryBuffer()
    for _ in range(NUM_PAST_STEPS):
        buf_b.update([state_b])
    out_b = build_neighbor_agents_past(buf_b, anchor)
    row_b = next((out_b[i] for i in range(NUM_AGENTS) if not np.all(out_b[i] == 0)), None)
    if row_b is None:
        print("  [FAIL] sub-case B: no non-zero row"); return False
    lb = row_b[-1]
    ok_b = check(abs(lb[4]) < 0.1 and abs(lb[5] + 1.0) < 0.1,
                 "ped crossing (+X CARLA) → ego vx≈0, vy≈-1",
                 f"got vx={lb[4]:.3f} vy={lb[5]:.3f}")
    all_ok = all_ok and ok_b

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    results = [
        test_empty_history(),
        test_single_vehicle_ahead(),
        test_agent_type_encoding(),
        test_history_padding(),
        test_lateral_placement(),
        test_rotated_ego_heading(),
        test_lateral_velocity(),
        test_carla_passthrough_conversion(),
        test_carla_actor_relative_position(),
        test_carla_pedestrian_velocity(),
    ]

    passed = sum(bool(r) for r in results)
    total  = len(results)
    print(f"\n{'='*55}")
    print(f"Results: {passed}/{total} test groups passed")
    if passed < total:
        print("Some tests failed — check output above for details.")
    else:
        print("All tests passed — agent pipeline is working correctly.")
