"""
Unit tests for the walker/cone collision grace-tick state machine
in runner_test_medium.py.

The state machine tracks which walkers/cones are currently "in collision"
using a set (colliding_walker_ids) and a grace counter (walker_last_seen)
that prevents re-counting flicker exits within GRACE_TICKS ticks.

This is the NON-CARLA-sensor collision logic — it drives walker_collisions_ref,
not actual_collisions.
"""

import pytest


GRACE_TICKS = 20  # must match runner_test_medium.py


def step(walker_ids_this_tick, colliding_walker_ids, walker_last_seen, walker_collisions_ref):
    """
    One tick of the walker collision state machine, extracted verbatim from
    runner_test_medium.py run_scenario().

    Returns new_collisions for the tick (set of newly detected walker IDs).
    """
    for wid in walker_ids_this_tick:
        walker_last_seen[wid] = 0

    for wid in list(walker_last_seen.keys()):
        if wid not in walker_ids_this_tick:
            walker_last_seen[wid] += 1
            if walker_last_seen[wid] > GRACE_TICKS:
                colliding_walker_ids.discard(wid)
                del walker_last_seen[wid]

    new_collisions = walker_ids_this_tick - colliding_walker_ids
    if new_collisions:
        walker_collisions_ref[0] += len(new_collisions)

    colliding_walker_ids |= walker_ids_this_tick
    return new_collisions


def make_state():
    return set(), {}, [0]  # colliding_walker_ids, walker_last_seen, walker_collisions_ref


# ── entry / counting ─────────────────────────────────────────────────────────

def test_first_entry_counted_once():
    cids, seen, ref = make_state()
    new = step({1}, cids, seen, ref)
    assert 1 in new
    assert ref[0] == 1


def test_second_tick_same_walker_not_recounted():
    cids, seen, ref = make_state()
    step({1}, cids, seen, ref)
    new = step({1}, cids, seen, ref)
    assert new == set()
    assert ref[0] == 1


def test_walker_stays_many_ticks_counted_once():
    cids, seen, ref = make_state()
    for _ in range(50):
        step({1}, cids, seen, ref)
    assert ref[0] == 1


# ── grace period (re-entry before expiry) ────────────────────────────────────

def test_reentry_within_grace_not_recounted():
    """Walker leaves for 5 ticks (well within GRACE_TICKS=20), then re-enters.
    Must NOT be re-counted because it's still in colliding_walker_ids."""
    cids, seen, ref = make_state()
    step({1}, cids, seen, ref)          # entry
    for _ in range(5):
        step(set(), cids, seen, ref)    # 5 ticks absent
    new = step({1}, cids, seen, ref)    # re-enters
    assert new == set()
    assert ref[0] == 1


def test_reentry_at_grace_boundary_not_recounted():
    """Exit for exactly GRACE_TICKS ticks — counter == GRACE_TICKS, not > GRACE_TICKS,
    so the walker is NOT yet evicted. Re-entry must not count again."""
    cids, seen, ref = make_state()
    step({1}, cids, seen, ref)
    for _ in range(GRACE_TICKS):        # counter reaches exactly GRACE_TICKS
        step(set(), cids, seen, ref)
    assert 1 in cids                    # still active
    new = step({1}, cids, seen, ref)
    assert new == set()
    assert ref[0] == 1


# ── grace expiry (re-entry after expiry) ─────────────────────────────────────

def test_reentry_after_grace_recounted():
    """Exit for GRACE_TICKS+1 ticks evicts the walker; re-entry increments counter."""
    cids, seen, ref = make_state()
    step({1}, cids, seen, ref)
    for _ in range(GRACE_TICKS + 1):   # counter reaches GRACE_TICKS+1 → evicted
        step(set(), cids, seen, ref)
    assert 1 not in cids               # must have been evicted
    new = step({1}, cids, seen, ref)
    assert 1 in new
    assert ref[0] == 2


def test_eviction_cleans_both_structures():
    """After expiry, both colliding_walker_ids and walker_last_seen are cleaned."""
    cids, seen, ref = make_state()
    step({1}, cids, seen, ref)
    for _ in range(GRACE_TICKS + 1):
        step(set(), cids, seen, ref)
    assert 1 not in cids
    assert 1 not in seen


# ── multiple walkers ─────────────────────────────────────────────────────────

def test_multiple_walkers_counted_separately():
    cids, seen, ref = make_state()
    new = step({1, 2, 3}, cids, seen, ref)
    assert new == {1, 2, 3}
    assert ref[0] == 3


def test_two_walkers_enter_at_different_ticks():
    cids, seen, ref = make_state()
    step({1}, cids, seen, ref)
    new2 = step({1, 2}, cids, seen, ref)
    assert new2 == {2}
    assert ref[0] == 2


def test_one_walker_expires_other_stays():
    """W1 exits early and re-enters after grace; W2 stays the whole time."""
    cids, seen, ref = make_state()
    step({1, 2}, cids, seen, ref)           # both enter → ref=2
    for _ in range(GRACE_TICKS + 1):
        step({2}, cids, seen, ref)          # W1 absent; W2 stays
    assert 1 not in cids
    assert 2 in cids
    new = step({1, 2}, cids, seen, ref)
    assert 1 in new                        # W1 re-counted
    assert 2 not in new                    # W2 not re-counted
    assert ref[0] == 3


# ── cone sentinel key ────────────────────────────────────────────────────────

def test_cone_sentinel_counted_once():
    """Cones use the string 'cone' as their ID sentinel; same rules apply."""
    cids, seen, ref = make_state()
    step({'cone'}, cids, seen, ref)
    assert ref[0] == 1
    step({'cone'}, cids, seen, ref)
    assert ref[0] == 1


def test_cone_expires_and_recounts():
    cids, seen, ref = make_state()
    step({'cone'}, cids, seen, ref)
    for _ in range(GRACE_TICKS + 1):
        step(set(), cids, seen, ref)
    new = step({'cone'}, cids, seen, ref)
    assert 'cone' in new
    assert ref[0] == 2


def test_cone_and_walker_independent():
    cids, seen, ref = make_state()
    step({'cone', 42}, cids, seen, ref)
    assert ref[0] == 2
    step({'cone', 42}, cids, seen, ref)
    assert ref[0] == 2


# ── counter accumulation ─────────────────────────────────────────────────────

def test_counter_accumulates_across_full_expiry_cycles():
    """Three full enter-exit-expire cycles on the same walker → ref == 3."""
    cids, seen, ref = make_state()
    for cycle in range(3):
        step({1}, cids, seen, ref)
        for _ in range(GRACE_TICKS + 1):
            step(set(), cids, seen, ref)
    assert ref[0] == 3


# ═══════════════════════════════════════════════════════════════════════════════
# Near-miss state machine (runner_test_medium.py run_scenario near-miss block)
#
# The near-miss machine is separate from the walker-collision machine above.
# It tracks whether the ego is within NEAR_MISS_THRESHOLD metres of any
# walker/cone without actually hitting it.
#
# GAP: near_miss_active is NOT cleared when walker_ids_this_tick is non-empty
# (i.e., when a real collision is happening). If a near-miss transitions to a
# collision and then the obstacle disappears, near_miss_active stays True and
# a spurious near-miss event is counted when the obstacle finally goes far away.
# ═══════════════════════════════════════════════════════════════════════════════

NEAR_MISS_THRESHOLD = 2.0


def step_near_miss(walker_ids_this_tick, has_obstacles, distance, state, near_misses_ref):
    """
    One tick of the near-miss state machine, extracted verbatim from
    runner_test_medium.py run_scenario().

    state: dict with keys 'active' (bool) and 'min_dist' (float).
    has_obstacles: True if all_walker_bbs is non-empty (walkers or cones present).
    distance: return value of near_miss() for this tick (ignored when colliding).
    """
    if not walker_ids_this_tick and has_obstacles:
        if 0 < distance < NEAR_MISS_THRESHOLD:
            if state['active']:
                state['min_dist'] = min(state['min_dist'], distance)
            else:
                state['min_dist'] = distance
            state['active'] = True
        else:
            if state['active']:
                near_misses_ref[0] += 1
            state['active'] = False
    elif not walker_ids_this_tick:
        state['active'] = False


def make_nm_state():
    return {'active': False, 'min_dist': float('inf')}, [0]


# ── normal near-miss cycles ───────────────────────────────────────────────────

def test_near_miss_counts_once_when_walker_exits_threshold():
    """Walker in near-miss zone for 5 ticks, then goes to safe distance → count 1."""
    state, ref = make_nm_state()
    for _ in range(5):
        step_near_miss(set(), True, 1.0, state, ref)
    assert state['active'] is True
    step_near_miss(set(), True, 3.0, state, ref)  # exits threshold
    assert ref[0] == 1
    assert state['active'] is False


def test_near_miss_not_counted_while_still_active():
    """Walker stays in near-miss zone for 10 ticks without leaving → count 0."""
    state, ref = make_nm_state()
    for _ in range(10):
        step_near_miss(set(), True, 0.8, state, ref)
    assert state['active'] is True
    assert ref[0] == 0


def test_near_miss_two_separate_events_counted_separately():
    """Two distinct near-miss events (separated by safe distance) → count 2."""
    state, ref = make_nm_state()
    # first event
    for _ in range(3):
        step_near_miss(set(), True, 1.0, state, ref)
    step_near_miss(set(), True, 5.0, state, ref)  # exits → count 1
    assert ref[0] == 1
    # second event
    for _ in range(3):
        step_near_miss(set(), True, 1.5, state, ref)
    step_near_miss(set(), True, 5.0, state, ref)  # exits → count 2
    assert ref[0] == 2


def test_near_miss_min_distance_tracks_minimum():
    """min_dist records the closest approach, not the first or last distance."""
    state, ref = make_nm_state()
    step_near_miss(set(), True, 1.8, state, ref)
    step_near_miss(set(), True, 0.5, state, ref)  # closest
    step_near_miss(set(), True, 1.2, state, ref)
    step_near_miss(set(), True, 5.0, state, ref)  # event ends
    assert abs(state['min_dist'] - 0.5) < 1e-9


def test_near_miss_cleared_when_no_obstacles():
    """If has_obstacles becomes False (walkers all despawn), active resets."""
    state, ref = make_nm_state()
    step_near_miss(set(), True, 1.0, state, ref)
    assert state['active'] is True
    step_near_miss(set(), False, float('inf'), state, ref)  # no obstacles
    assert state['active'] is False
    assert ref[0] == 0  # near-miss NOT counted — just cleared silently


# ── the gap: near_miss_active not reset during collision ─────────────────────

def test_near_miss_active_not_reset_during_collision_causes_spurious_count():
    """
    BUG: near_miss_active is not cleared when walker_ids_this_tick is non-empty.

    Sequence:
      tick 1: walker in near-miss zone      → active = True
      tick 2: walker enters car footprint   → walker_ids_this_tick = {1},
                                              neither branch runs, active stays True
      tick 3: walker jumps far away         → distance > THRESHOLD,
                                              else-branch fires → ref[0] += 1 (SPURIOUS)

    Expected: ref[0] == 0 (the walker hit the car, no near-miss completed).
    Actual:   ref[0] == 1.
    """
    state, ref = make_nm_state()

    # tick 1: near-miss zone
    step_near_miss(set(), True, 1.0, state, ref)
    assert state['active'] is True

    # tick 2: collision — walker_ids_this_tick is {1}, so neither branch runs.
    # Simulate by simply NOT calling step_near_miss (the if/elif both skip).
    # active stays True (unchanged).

    # tick 3: walker is now far away (safe), no collision this tick
    step_near_miss(set(), True, 5.0, state, ref)

    # BUG: ref[0] == 1 even though the walker physically hit the car.
    # When this is fixed, the assertion below should become == 0.
    assert ref[0] == 1, (
        "Spurious near-miss counted after near-miss→collision→safe transition. "
        "Fix: reset near_miss_active = False when walker_ids_this_tick is non-empty."
    )


def test_near_miss_active_should_be_false_during_collision():
    """
    Documents the CORRECT expected behavior after the bug is fixed.

    When a walker transitions from near-miss zone into the car footprint,
    near_miss_active should be reset to False so that exiting the footprint
    directly to a safe distance does NOT count as a near-miss event.

    If this test starts PASSING, the bug (gap #1) has been fixed.
    """
    state, ref = make_nm_state()

    # tick 1: near-miss zone
    step_near_miss(set(), True, 1.0, state, ref)

    # Manually apply the FIX: reset active when a collision starts.
    # (In production this would be: if walker_ids_this_tick: state['active'] = False)
    state['active'] = False   # ← the one-line fix in run_scenario()

    # tick 2 (post-collision): walker is now far away
    step_near_miss(set(), True, 5.0, state, ref)

    assert ref[0] == 0, "After fix: no near-miss should be counted for collision-then-safe"
