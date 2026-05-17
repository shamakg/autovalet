"""
Integration tests for cone spawn correctness and lifecycle.

A typo in parking_cone.py (`self.world.tick()` instead of `self._world.tick()`)
caused an AttributeError that was silently caught, leaving every cone:
  - not added to `self.cones` or `self.other_actors`  →  cleanup() never destroyed them
  - with simulate_physics=True                         →  could be knocked over
  - only `cone_2.set_simulate_physics(False)` was missing (only cone, not cone_2)

Cones were still visible to get_bounding_boxes() (it queries world.get_actors()
directly), so collision detection *worked*, but actors accumulated across runs.

Tests here verify:
  1. Cones are tracked in `ParkingConeScenario.cones` (not leaked)
  2. Actor count in world matches tracked count (no orphans)
  3. After scenario.cleanup(), all cone actors are gone from the world
  4. Cones land near terrain level (physics snap during the tick works correctly)

Run with:
    cd ~/carla_garage/leaderboard/leaderboard/autovalet
    source ~/envs/simlingo/bin/activate
    python -m pytest testbed_tests/test_integration_cone_spawn.py -v -s
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import carla
import pytest

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration

pytestmark = pytest.mark.integration


def make_config():
    cfg = ScenarioConfiguration()
    cfg.name = "ConeSpawnTest"; cfg.type = "Parking"
    cfg.town = "Town04_Opt"; cfg.other_actors = None; cfg.route = False
    return cfg


def cleanup(scenario, world):
    if scenario:
        try:
            scenario.cleanup()
        except Exception as e:
            print("cleanup: %s" % e)
    for _ in range(3):
        world.tick()


def cone_count_in_world(world):
    """Count construction-cone props currently in the world."""
    return sum(1 for a in world.get_actors().filter('static.prop.*')
               if 'cone' in a.type_id.lower())


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Cones are registered in ParkingConeScenario.cones (not orphaned)
# ═══════════════════════════════════════════════════════════════════════════════

def test_cones_tracked_in_scenario(carla_world):
    """
    After ParkingScenarioEasy spawns cones, ParkingConeScenario.cones must
    contain the spawned actors.  The old bug (self.world.tick() AttributeError)
    left the list empty so scenario.cleanup() never destroyed the actors.
    """
    from parking_scenarios.parking_scenario_easy import ParkingScenarioEasy

    scenario = None
    world = carla_world
    try:
        scenario = ParkingScenarioEasy(
            world=world, config=make_config(),
            destination=20, parked=[],
            criteria_enable=False,
        )
        world.tick()

        cone_scenario = scenario.list_scenarios[0]
        assert len(cone_scenario.cones) > 0, (
            "ParkingConeScenario.cones is empty — the self.world.tick() typo "
            "fix may not be in effect, or no empty spots were available"
        )

    finally:
        cleanup(scenario, world)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Tracked cones match world actor count (no orphans)
# ═══════════════════════════════════════════════════════════════════════════════

def test_no_orphaned_cone_actors(carla_world):
    """
    The number of construction-cone props visible in world.get_actors() must
    equal the number of cones in ParkingConeScenario.cones.  Any excess means
    cones were spawned but not tracked — they'll outlive the scenario.
    """
    from parking_scenarios.parking_scenario_easy import ParkingScenarioEasy

    scenario = None
    world = carla_world
    # Measure baseline cones before scenario (from previous tests)
    baseline = cone_count_in_world(world)

    try:
        scenario = ParkingScenarioEasy(
            world=world, config=make_config(),
            destination=20, parked=[],
            criteria_enable=False,
        )
        world.tick()

        cone_scenario = scenario.list_scenarios[0]
        tracked = len(cone_scenario.cones)
        in_world = cone_count_in_world(world) - baseline

        assert tracked == in_world, (
            "Cone count mismatch: %d tracked in scenario.cones but %d in world. "
            "%d orphaned cones will never be cleaned up." % (
                tracked, in_world, in_world - tracked
            )
        )

    finally:
        cleanup(scenario, world)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Cleanup removes all cone actors from the world
# ═══════════════════════════════════════════════════════════════════════════════

def test_cleanup_destroys_all_cones(carla_world):
    """
    After scenario.cleanup(), no construction-cone props from this scenario
    should remain in the world.  This verifies the full lifecycle: spawn →
    track → destroy.
    """
    from parking_scenarios.parking_scenario_easy import ParkingScenarioEasy

    world = carla_world
    baseline = cone_count_in_world(world)

    scenario = ParkingScenarioEasy(
        world=world, config=make_config(),
        destination=20, parked=[],
        criteria_enable=False,
    )
    world.tick()

    cones_spawned = cone_count_in_world(world) - baseline
    assert cones_spawned > 0, "No cones were spawned — test can't verify cleanup"

    cleanup(scenario, world)

    remaining = cone_count_in_world(world) - baseline
    assert remaining == 0, (
        "%d cone actor(s) survived scenario.cleanup() — actor leak detected. "
        "These accumulate across benchmark runs." % remaining
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Cones land near terrain level after physics snap
# ═══════════════════════════════════════════════════════════════════════════════

def test_cone_z_near_terrain_after_spawn(carla_world):
    """
    Cones are placed via ground_projection + BB half-height so their base sits
    on the terrain surface.  Static props don't respond to gravity, so the old
    approach of spawning at z=1.5 and relying on a physics-snap tick left every
    cone floating 1.3 m in the air.

    After the fix the cone origin should be < 1.0 m (terrain z≈0.15 + cone
    half-height ≈ 0.25 ⇒ origin ≈ 0.4 m).  z=1.5 would clearly fail.
    """
    from parking_scenarios.parking_scenario_easy import ParkingScenarioEasy

    scenario = None
    world = carla_world
    try:
        scenario = ParkingScenarioEasy(
            world=world, config=make_config(),
            destination=20, parked=[],
            criteria_enable=False,
        )
        world.tick()

        cone_scenario = scenario.list_scenarios[0]
        if len(cone_scenario.cones) == 0:
            pytest.skip("No cones spawned")

        bad_z = []
        for cone in cone_scenario.cones:
            if not cone.is_alive:
                continue
            loc = cone.get_location()
            # terrain_z≈0.15 + bb_half_h≈0.25 → origin≈0.4 m.
            # z > 1.0: ground_projection fix not in effect (still spawned at z=1.5).
            # z <= 0.0: cone landed at world origin (set_location failed).
            if loc.z > 1.0:
                bad_z.append(
                    "cone %d at (%.1f, %.1f): z=%.3f — floating (expected 0 < z <= 1.0)"
                    % (cone.id, loc.x, loc.y, loc.z)
                )
            elif loc.z <= 0.0:
                bad_z.append(
                    "cone %d at (%.1f, %.1f): z=%.3f — at world origin (set_location may have failed)"
                    % (cone.id, loc.x, loc.y, loc.z)
                )

        assert not bad_z, (
            "Cone(s) have unexpected z — ground_projection placement fix "
            "may not be active in parking_cone.py:\n" + "\n".join(bad_z)
        )

    finally:
        cleanup(scenario, world)
