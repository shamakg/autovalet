"""
Integration tests for pedestrian spawn quality — z-coordinate / ground-clipping.

The user observed walkers appearing "halfway in the ground".  These tests probe
every stage where z can go wrong:

  1. After _initialize_actors / _replace_walker, walker FEET (BB bottom) must be
     above terrain level.  CARLA walker BB half-height ≈ 0.93 m; origin at z=0.5
     puts feet at z=−0.43, visibly underground.  Fix: terrain_z + bb_half_h + 0.1.

  2. After MoveWalkerNoPhysics runs (x-axis movement toward lane edge), z must
     stay at the spawn value — the behavior updates only x/y, not z.

  3. After the behavior tree fires ActorTransformSetter (re-positions walker to
     visible spawn point), feet must still be above terrain.

  4. Walker BB height must be in the human range [0.4 m, 2.0 m].

  5. _replace_walker must leave simulate_physics=False — z must not drift across
     world ticks.

Ground truth: in Town04 parking lot terrain is z≈0.15–0.20.  We use
  feet_z(actor) >= 0.0  as a simple, reliable ground-above check that is
  immune to ground_projection returning parked-car roofs (z≈1.4) due to
  leftover actors from earlier tests in the same session.

Run with:
    cd ~/carla_garage/leaderboard/leaderboard/autovalet
    source ~/envs/simlingo/bin/activate
    python -m pytest testbed_tests/test_integration_pedestrian_spawn.py -v -s
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import carla
import pytest

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration

pytestmark = pytest.mark.integration

# BB bottom must be >= this z.  Town04 parking lot terrain is z≈0.15, so 0.0
# is a safe conservative floor — any negative value means visibly underground.
FEET_Z_MIN = 0.0


def make_config():
    cfg = ScenarioConfiguration()
    cfg.name = "PedSpawnTest"; cfg.type = "Parking"
    cfg.town = "Town04_Opt"; cfg.other_actors = None; cfg.route = False
    return cfg


def feet_z(actor):
    """Return the z coordinate of the walker's lowest BB point (its feet)."""
    loc = actor.get_location()
    bb  = actor.bounding_box
    return loc.z + bb.location.z - bb.extent.z


def cleanup(scenario, world):
    if scenario:
        try:
            scenario.cleanup()
        except Exception as e:
            print("cleanup: %s" % e)
    for _ in range(3):
        world.tick()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Walker feet above terrain right after _initialize_actors
# ═══════════════════════════════════════════════════════════════════════════════

def test_walker_feet_above_ground_after_init(carla_world):
    """
    Right after PedestrianCrossingParking._initialize_actors, every walker's
    BB bottom (feet) must be above z=0.

    Old bug: _replace_walker set origin z=0.5 but walker BB half-height=0.93 m,
    so feet were at z=0.5−0.93=−0.43 (nearly half a metre underground).

    Fix: _replace_walker now uses terrain_z + bb_half_height + 0.1 so feet land
    at terrain_z + 0.1 m above the ground surface.
    """
    from parking_scenarios.parking_scenario_hard import ParkingScenarioHard, HardMode

    scenario = None
    world = carla_world
    try:
        scenario = ParkingScenarioHard(
            world=world, config=make_config(),
            destination=20, parked=[],
            criteria_enable=False, mode=HardMode.PedMode,
        )
        world.tick()

        ped_scenario = scenario.list_scenarios[0]
        assert len(ped_scenario.other_actors) > 0, "no walkers spawned"

        underground = []
        for i, actor in enumerate(ped_scenario.other_actors):
            if 'walker' not in actor.type_id or 'controller' in actor.type_id:
                continue
            fz = feet_z(actor)
            if fz < FEET_Z_MIN:
                underground.append(
                    "walker %d at (%.2f, %.2f, %.2f): feet_z=%.3f (must be >= %.1f)"
                    % (i, actor.get_location().x, actor.get_location().y,
                       actor.get_location().z, fz, FEET_Z_MIN)
                )

        assert not underground, (
            "Walker feet are below z=0 — visibly underground:\n"
            + "\n".join(underground)
        )

    finally:
        cleanup(scenario, world)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Spawn z is dynamically computed (terrain-based), not hardcoded 0.5
# ═══════════════════════════════════════════════════════════════════════════════

def test_spawn_z_uses_terrain_plus_bb_offset(carla_world):
    """
    _replace_walker() must set origin z = terrain_z + bb_half_height + 0.1
    so feet are clear of terrain.  This replaces the old hardcoded z=0.5.

    Verify: for every spawned walker,
       origin_z  >  0.5    (strict: must exceed the old hardcoded value)
       feet_z    >= 0.0    (feet above z=0, which is below all terrain here)

    Since Town04 parking lot terrain is z≈0.15–0.20 and bb_half_height≈0.93 m,
    the correct origin is ≈1.2 m.  Anything significantly higher than 0.5
    indicates the fix is active.
    """
    from parking_scenarios.parking_scenario_hard import ParkingScenarioHard, HardMode

    scenario = None
    world = carla_world
    try:
        scenario = ParkingScenarioHard(
            world=world, config=make_config(),
            destination=15, parked=[],
            criteria_enable=False, mode=HardMode.PedMode,
        )
        world.tick()

        ped_scenario = scenario.list_scenarios[0]
        walkers = [a for a in ped_scenario.other_actors
                   if 'walker' in a.type_id and 'controller' not in a.type_id]
        assert walkers, "no walkers spawned"

        still_hardcoded = []
        for actor in walkers:
            origin_z = actor.get_location().z
            fz = feet_z(actor)
            # If origin_z ≈ 0.5 the fix didn't run; correct value is ≈1.2
            if origin_z <= 0.6:
                still_hardcoded.append(
                    "walker %d: origin_z=%.3f (expected > 0.6); feet_z=%.3f"
                    % (actor.id, origin_z, fz)
                )
            elif fz < FEET_Z_MIN:
                still_hardcoded.append(
                    "walker %d: origin_z=%.3f but feet_z=%.3f < 0 — still underground"
                    % (actor.id, origin_z, fz)
                )

        assert not still_hardcoded, (
            "Walker(s) still placed with hardcoded z=0.5 or still underground:\n"
            + "\n".join(still_hardcoded)
        )

    finally:
        cleanup(scenario, world)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Walker z does not change during MoveWalkerNoPhysics ticks
# ═══════════════════════════════════════════════════════════════════════════════

def test_walker_z_stable_during_move_no_physics(carla_world):
    """
    MoveWalkerNoPhysics reads get_location() and updates only x/y, not z.
    If z was correct on spawn it should stay correct throughout movement.

    A drift > 0.1 m downward from spawn z indicates the actor is sinking.
    """
    from parking_scenarios.parking_scenario_hard import ParkingScenarioHard, HardMode

    scenario = None
    world = carla_world
    try:
        scenario = ParkingScenarioHard(
            world=world, config=make_config(),
            destination=20, parked=[],
            criteria_enable=False, mode=HardMode.PedMode,
        )
        world.tick()

        ped_scenario = scenario.list_scenarios[0]
        assert len(ped_scenario.other_actors) > 0, "no walkers spawned"

        spawn_z = {}
        for actor in ped_scenario.other_actors:
            if 'walker' in actor.type_id and 'controller' not in actor.type_id:
                spawn_z[actor.id] = actor.get_location().z

        # Teleport ego within trigger distance of first walker
        ego_loc = scenario.car.actor.get_location()
        for wp_data in ped_scenario._walker_data:
            trigger_loc = wp_data['transform'].location
            scenario.car.actor.set_location(
                carla.Location(x=trigger_loc.x, y=trigger_loc.y - 5.0, z=ego_loc.z)
            )
            break

        scenario_tree = ped_scenario.scenario_tree
        sinking = []
        for tick_i in range(30):
            scenario_tree.tick_once()
            world.tick()

            for actor in ped_scenario.other_actors:
                if not actor.is_alive:
                    continue
                if 'walker' not in actor.type_id or 'controller' in actor.type_id:
                    continue
                sz = spawn_z.get(actor.id, 1.0)
                cur_z = actor.get_location().z
                if cur_z < sz - 0.1:
                    sinking.append(
                        "tick %d walker %d: spawn_z=%.3f, current_z=%.3f, "
                        "sank %.3f m" % (tick_i, actor.id, sz, cur_z, sz - cur_z)
                    )

        assert not sinking, (
            "Walker(s) sank below their spawn z during MoveWalkerNoPhysics:\n"
            + "\n".join(sinking)
        )

    finally:
        cleanup(scenario, world)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Walker feet above ground after ActorTransformSetter repositions to visible point
# ═══════════════════════════════════════════════════════════════════════════════

def test_walker_feet_above_ground_after_actor_transform_setter(carla_world):
    """
    The behavior tree runs ActorTransformSetter(walker, spawn_transform, False)
    which moves the walker from its hidden underground position to the visible
    spawn point stored in walker_data['transform'].

    Since walker_data['transform'].location.z is set by _replace_walker to
    terrain_z + bb_half_h + 0.1, after ActorTransformSetter fires, feet must
    remain above z=0.
    """
    from parking_scenarios.parking_scenario_hard import ParkingScenarioHard, HardMode

    scenario = None
    world = carla_world
    try:
        scenario = ParkingScenarioHard(
            world=world, config=make_config(),
            destination=25, parked=[],
            criteria_enable=False, mode=HardMode.PedMode,
        )
        world.tick()

        ped_scenario = scenario.list_scenarios[0]
        assert len(ped_scenario.other_actors) > 0, "no walkers spawned"

        scenario_tree = ped_scenario.scenario_tree
        for _ in range(5):
            scenario_tree.tick_once()
            world.tick()

        underground = []
        for i, actor in enumerate(ped_scenario.other_actors):
            if not actor.is_alive:
                continue
            if 'walker' not in actor.type_id or 'controller' in actor.type_id:
                continue
            fz = feet_z(actor)
            if fz < FEET_Z_MIN:
                underground.append(
                    "walker %d: origin_z=%.3f, feet_z=%.3f (must be >= %.1f)"
                    % (i, actor.get_location().z, fz, FEET_Z_MIN)
                )

        assert not underground, (
            "Walker feet underground after ActorTransformSetter — "
            "spawn_transform.location.z may be wrong:\n" + "\n".join(underground)
        )

    finally:
        cleanup(scenario, world)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Walker bounding box has positive height (not collapsed underground)
# ═══════════════════════════════════════════════════════════════════════════════

def test_walker_bounding_box_height_reasonable(carla_world):
    """
    A walker clipped halfway into the ground would have its bounding box origin
    at ground level (height ≈ half normal) or could cause degenerate BB geometry.

    This test checks that each walker's bounding box extent in z is within
    [0.4 m, 2.0 m] — CARLA pedestrian models range up to ~1.9 m.  A value
    near 0 means the BB collapsed (walker fully underground); a value >> 2.0
    suggests something is very wrong.
    """
    from parking_scenarios.parking_scenario_hard import ParkingScenarioHard, HardMode

    scenario = None
    world = carla_world
    try:
        scenario = ParkingScenarioHard(
            world=world, config=make_config(),
            destination=30, parked=[],
            criteria_enable=False, mode=HardMode.PedMode,
        )
        world.tick()

        ped_scenario = scenario.list_scenarios[0]
        assert len(ped_scenario.other_actors) > 0, "no walkers spawned"

        bad_heights = []
        for actor in ped_scenario.other_actors:
            if 'walker' not in actor.type_id or 'controller' in actor.type_id:
                continue
            bb = actor.bounding_box
            height_m = bb.extent.z * 2  # full height = 2 × half-extent
            if not (0.4 <= height_m <= 2.0):  # CARLA pedestrians range up to ~1.9 m
                bad_heights.append(
                    "walker %d (type=%s): BB z-extent=%.3f, full height=%.3f m"
                    % (actor.id, actor.type_id, bb.extent.z, height_m)
                )

        assert not bad_heights, (
            "Walker(s) have unexpected bounding box height — may indicate "
            "underground clipping:\n" + "\n".join(bad_heights)
        )

    finally:
        cleanup(scenario, world)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. _replace_walker leaves simulate_physics=False (z must not drift)
# ═══════════════════════════════════════════════════════════════════════════════

def test_replace_walker_physics_off_and_z_preserved(carla_world):
    """
    _replace_walker() calls set_simulate_physics(False).  After the next
    world.tick(), CARLA must not move the actor vertically.  If physics were
    accidentally left on, the walker would fall/rise (z changing by > 0.1 m).
    """
    from parking_scenarios.parking_scenario_hard import ParkingScenarioHard, HardMode

    scenario = None
    world = carla_world
    try:
        scenario = ParkingScenarioHard(
            world=world, config=make_config(),
            destination=15, parked=[],
            criteria_enable=False, mode=HardMode.PedMode,
        )
        world.tick()

        ped_scenario = scenario.list_scenarios[0]
        walkers = [a for a in ped_scenario.other_actors
                   if 'walker' in a.type_id and 'controller' not in a.type_id]
        assert walkers, "no walkers spawned"

        z_before = {a.id: a.get_location().z for a in walkers}

        for _ in range(3):
            world.tick()

        z_drifted = []
        for actor in walkers:
            if not actor.is_alive:
                continue
            z_now = actor.get_location().z
            z_was = z_before[actor.id]
            if abs(z_now - z_was) > 0.05:
                z_drifted.append(
                    "walker %d: z before=%.3f, after 3 ticks=%.3f, drift=%.3f m"
                    % (actor.id, z_was, z_now, z_now - z_was)
                )

        assert not z_drifted, (
            "Walker z changed across ticks — simulate_physics may not be False:\n"
            + "\n".join(z_drifted)
        )

    finally:
        cleanup(scenario, world)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Walkers from different destinations all have feet above ground
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("destination", [10, 20, 30])
def test_walker_feet_above_ground_multiple_destinations(carla_world, destination):
    """
    Parametrize over several destinations to ensure no specific parking spot
    geometry causes walkers to spawn with feet underground.  Each destination
    uses different lane_waypoints which map to different x/y spawn positions.
    """
    from parking_scenarios.parking_scenario_hard import ParkingScenarioHard, HardMode

    scenario = None
    world = carla_world
    try:
        scenario = ParkingScenarioHard(
            world=world, config=make_config(),
            destination=destination, parked=[],
            criteria_enable=False, mode=HardMode.PedMode,
        )
        world.tick()

        ped_scenario = scenario.list_scenarios[0]
        if len(ped_scenario.other_actors) == 0:
            pytest.skip("No walkers spawned for destination %d" % destination)

        underground = []
        for i, actor in enumerate(ped_scenario.other_actors):
            if not actor.is_alive:
                continue
            if 'walker' not in actor.type_id or 'controller' in actor.type_id:
                continue
            fz = feet_z(actor)
            if fz < FEET_Z_MIN:
                underground.append(
                    "dest=%d walker %d (%.1f, %.1f): origin_z=%.3f, feet_z=%.3f"
                    % (destination, i,
                       actor.get_location().x, actor.get_location().y,
                       actor.get_location().z, fz)
                )

        assert not underground, (
            "Walker feet underground for destination=%d:\n" % destination
            + "\n".join(underground)
        )

    finally:
        cleanup(scenario, world)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. SmartPedestrianCrossingParking walker feet above ground
# ═══════════════════════════════════════════════════════════════════════════════

def _cleanup_smart(ped_scenario, ego_car, world):
    """Clean up a directly-instantiated SmartPedestrianCrossingParking."""
    if ped_scenario:
        try:
            ped_scenario.remove_all_actors()
        except Exception as e:
            print("smart cleanup remove_all_actors: %s" % e)
    if ego_car and ego_car.actor and ego_car.actor.is_alive:
        try:
            ego_car.actor.destroy()
        except Exception:
            pass
    for _ in range(3):
        world.tick()


@pytest.mark.parametrize("destination", [10, 20, 30])
def test_smart_pedestrian_feet_above_ground(carla_world, destination):
    """
    SmartPedestrianCrossingParking._replace_walker() had a hardcoded z=0.5 that
    put walker feet at z=−0.43 m (underground).  After the ground_projection fix,
    every walker's feet (BB bottom) must be at or above z=0.

    Covers the same ground as test_walker_feet_above_ground_multiple_destinations
    but exercises SmartPedestrianCrossingParking directly rather than through
    ParkingScenarioHard (which uses PedestrianCrossingParking instead).
    """
    from parking_scenarios.smart_pedestrian_crossing_parking import SmartPedestrianCrossingParking
    from testbed.v2_experiment_utils import town04_spawn_ego_vehicle
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

    world = carla_world
    ego_car = None
    ped_scenario = None
    try:
        ego_car = town04_spawn_ego_vehicle(world, destination)
        CarlaDataProvider.register_actor(ego_car.actor)
        world.tick()

        cfg = make_config()
        cfg.name = "SmartPedSpawnTest"
        cfg.type = "PedestrianCrossing"

        ped_scenario = SmartPedestrianCrossingParking(
            world=world,
            ego_vehicles=[ego_car.actor],
            config=cfg,
            parking_spot=destination,
            criteria_enable=False,
        )
        world.tick()

        walkers = [
            a for a in ped_scenario.other_actors
            if 'walker' in a.type_id and 'controller' not in a.type_id
        ]
        if not walkers:
            pytest.skip("No SmartPedestrian walkers spawned for destination %d" % destination)

        underground = []
        for actor in walkers:
            if not actor.is_alive:
                continue
            fz = feet_z(actor)
            if fz < FEET_Z_MIN:
                underground.append(
                    "smart dest=%d walker %d (%.1f, %.1f): origin_z=%.3f, feet_z=%.3f"
                    % (destination, actor.id,
                       actor.get_location().x, actor.get_location().y,
                       actor.get_location().z, fz)
                )

        assert not underground, (
            "SmartPedestrian walker feet underground for destination=%d — "
            "ground_projection fix may not be active in _replace_walker:\n" % destination
            + "\n".join(underground)
        )

    finally:
        _cleanup_smart(ped_scenario, ego_car, world)
