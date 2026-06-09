"""
Unit tests for OppositeDirectionVehicle spawn-location logic.

The three CollisionMode variants (COLLIDE, MISS, STOP_EARLY) each place
the oncoming vehicle at a different spawn_y relative to the ego/destination.
The logic is inlined here (mocking CARLA actor/location types) to be runnable
without a live server.

Key invariants tested:
  - COLLIDE: vehicle spawns far enough that trigger_distance == 2*offset
  - MISS:    spawn_y is shifted by miss_offset so the paths don't overlap
  - STOP_EARLY: drive_distance is reduced so the vehicle stops short of the ego
  - Lane selection: closest lane_x to ego is picked
"""

import pytest
import numpy as np
from enum import Enum


# ── minimal CARLA stubs ───────────────────────────────────────────────────────

class _Loc:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class _Actor:
    def __init__(self, x, y):
        self._loc = _Loc(x, y)
    def get_location(self):
        return self._loc


# ── inlined from opposite_vehicle_parking.py ────────────────────────────────

class CollisionMode(Enum):
    COLLIDE    = 0
    STOP_EARLY = 1
    MISS       = 2


def _get_vehicle_location(ego_x, ego_y, dest_x, dest_y, collision_mode,
                          lane_options, town04_y_max,
                          initial_speed, initial_trigger_distance,
                          lane_width=3, start_offset=25,
                          drive_distance=100, miss_offset=6, brake_speed=1.5):
    """
    Mirrors get_vehicle_location() from OppositeDirectionVehicle.
    Returns (spawn_x, spawn_y, trigger_distance, drive_distance, speed).
    """
    speed            = initial_speed
    trigger_distance = initial_trigger_distance

    lane_x   = min(lane_options, key=lambda x: abs(x - ego_x))
    dest_x_  = dest_x
    close_x  = lane_x + lane_width / 2

    if dest_x_ < lane_x:
        miss_offset *= -1.5

    if collision_mode == CollisionMode.COLLIDE:
        if dest_x_ > close_x:
            speed *= 0.8
        spawn_y        = min(town04_y_max, dest_y + abs(dest_y - ego_y))
        actual_offset  = abs(spawn_y - dest_y)
        trigger_distance = actual_offset * 2

    elif collision_mode == CollisionMode.MISS:
        spawn_y        = min(town04_y_max, dest_y + abs(dest_y - ego_y) - miss_offset)
        actual_offset  = abs(spawn_y - dest_y)
        trigger_distance = actual_offset * 2 + miss_offset

    elif collision_mode == CollisionMode.STOP_EARLY:
        if dest_x_ > close_x:
            speed *= 0.8
        spawn_y        = min(town04_y_max, dest_y + abs(dest_y - ego_y))
        actual_offset  = abs(spawn_y - dest_y)
        trigger_distance = actual_offset * 2
        drive_distance   = actual_offset - 7

    spawn_x = close_x
    return spawn_x, spawn_y, trigger_distance, drive_distance, speed


# ── shared test parameters ────────────────────────────────────────────────────

LANE_OPTIONS  = {280.0, 283.0, 286.0, 289.0}
TOWN04_Y_MAX  = -190.0   # Town04 upper bound (less negative = larger y)
BASE_SPEED    = 3.0
BASE_TRIGGER  = 40.0

# Typical scenario: ego at (286, -230), destination at (286, -215), same lane
EGO_X, EGO_Y  = 286.0, -230.0
DEST_X, DEST_Y = 286.0, -215.0


def run(mode, ego_x=EGO_X, ego_y=EGO_Y, dest_x=DEST_X, dest_y=DEST_Y):
    return _get_vehicle_location(
        ego_x, ego_y, dest_x, dest_y, mode,
        LANE_OPTIONS, TOWN04_Y_MAX,
        BASE_SPEED, BASE_TRIGGER,
    )


# ── lane selection ────────────────────────────────────────────────────────────

def test_closest_lane_selected():
    """Ego at x=285 should pick lane 286 (closest in LANE_OPTIONS)."""
    sx, _, _, _, _ = run(CollisionMode.COLLIDE, ego_x=285.0)
    # spawn_x = closest_lane + lane_width/2 = 286 + 1.5 = 287.5
    assert abs(sx - 287.5) < 1e-6


def test_different_ego_picks_different_lane():
    sx1, _, _, _, _ = run(CollisionMode.COLLIDE, ego_x=281.0)
    sx2, _, _, _, _ = run(CollisionMode.COLLIDE, ego_x=287.5)
    assert sx1 != sx2


# ── COLLIDE mode ──────────────────────────────────────────────────────────────

def test_collide_trigger_equals_twice_offset():
    """trigger_distance must equal 2 * actual_offset so the car arrives at dest."""
    _, spawn_y, trigger, _, _ = run(CollisionMode.COLLIDE)
    actual_offset = abs(spawn_y - DEST_Y)
    assert abs(trigger - 2 * actual_offset) < 1e-6


def test_collide_spawn_y_beyond_dest():
    """Oncoming car must start on the far side of the destination."""
    _, spawn_y, _, _, _ = run(CollisionMode.COLLIDE)
    # destination is at DEST_Y=-215, ego at EGO_Y=-230; spawn is more positive
    assert spawn_y >= DEST_Y


def test_collide_spawn_y_does_not_exceed_map():
    _, spawn_y, _, _, _ = run(CollisionMode.COLLIDE)
    assert spawn_y <= TOWN04_Y_MAX


# ── MISS mode ────────────────────────────────────────────────────────────────

def test_miss_spawn_y_shifted_by_miss_offset():
    """MISS spawn_y == COLLIDE spawn_y - miss_offset (right-side dest)."""
    _, collide_y, _, _, _ = run(CollisionMode.COLLIDE)
    _, miss_y,    _, _, _ = run(CollisionMode.MISS)
    # miss_offset=6 when dest is on the right (dest_x >= lane_x)
    assert abs(miss_y - (collide_y - 6)) < 1e-6


def test_miss_trigger_accounts_for_offset():
    """MISS trigger = 2*actual_offset + miss_offset."""
    _, spawn_y, trigger, _, _ = run(CollisionMode.MISS)
    actual_offset = abs(spawn_y - DEST_Y)
    expected_trigger = actual_offset * 2 + 6
    assert abs(trigger - expected_trigger) < 1e-6


def test_miss_spawn_y_closer_than_collide():
    """MISS vehicle starts closer to ego (less far) than COLLIDE vehicle."""
    _, collide_y, _, _, _ = run(CollisionMode.COLLIDE)
    _, miss_y,    _, _, _ = run(CollisionMode.MISS)
    # miss_y is 6 m closer (less positive)
    assert miss_y < collide_y


# ── STOP_EARLY mode ───────────────────────────────────────────────────────────

def test_stop_early_drive_distance_stops_short():
    """drive_distance = actual_offset - 7: vehicle stops 7 m before destination."""
    _, spawn_y, _, drive_dist, _ = run(CollisionMode.STOP_EARLY)
    actual_offset = abs(spawn_y - DEST_Y)
    assert abs(drive_dist - (actual_offset - 7)) < 1e-6


def test_stop_early_spawn_same_as_collide():
    """STOP_EARLY and COLLIDE start from the same spawn_y."""
    _, collide_y, _, _, _ = run(CollisionMode.COLLIDE)
    _, stop_y,    _, _, _ = run(CollisionMode.STOP_EARLY)
    assert abs(collide_y - stop_y) < 1e-6


def test_stop_early_drive_distance_positive():
    """drive_distance must be positive (actual_offset should be > 7)."""
    _, _, _, drive_dist, _ = run(CollisionMode.STOP_EARLY)
    assert drive_dist > 0, "drive_distance went negative — scenario geometry too tight"


# ── speed reduction ───────────────────────────────────────────────────────────

def test_collide_speed_reduced_when_dest_in_opposite_lane():
    """If destination is in the far lane (dest_x > close_x), speed is reduced 20%."""
    # Use dest_x far to the right so it's > close_x
    _, _, _, _, spd = run(CollisionMode.COLLIDE, dest_x=295.0)
    assert spd < BASE_SPEED


def test_collide_speed_unchanged_when_dest_in_ego_lane():
    """If destination is on the near side, speed is unchanged."""
    # dest_x == ego_x == 286, close_x = 287.5 → dest_x < close_x
    _, _, _, _, spd = run(CollisionMode.COLLIDE)
    assert abs(spd - BASE_SPEED) < 1e-6
