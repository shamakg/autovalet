"""
Integration tests for scenario-actor visibility and collision_tracker coverage.

These tests cover four gaps that the previous test files do NOT address:

  1. Scenario-spawned pedestrians (via CarlaDataProvider.request_new_actor)
     must appear in get_bounding_boxes() walker_bbs.  If they're invisible the
     benchmark silently misses ALL pedestrian collisions.

  2. The OppositeDirectionVehicle must appear in moving_cars_bbs (not filtered
     as ego or parked car) so the collision pipeline can see it.

  3. production builds collision_tracker = obstacle_map_from_bbs(all_bbs_list)
     with NO min_y.  If all actors are far from the ego start (y≈-243.8), the
     map may not extend there and generate_collision_mask returns zeros for the
     first N ticks — silent blind spot on the approach road.

  4. After ticking the scenario behavior tree, a moving pedestrian's updated
     position must still be tracked correctly by get_bounding_boxes().

  5. DoorMode (ParkingScenarioHard HardMode.DoorMode) spawning — actors alive.

Run with:
    cd ~/carla_garage/leaderboard/leaderboard/autovalet
    source ~/envs/simlingo/bin/activate
    python -m pytest testbed_tests/test_integration_scenario_actors.py -v -s
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import carla
import numpy as np
import pytest

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration

from parking_position import parking_vehicle_locations_Town04
from testbed.v2_experiment_utils import get_bounding_boxes, obstacle_map_from_bbs

pytestmark = pytest.mark.integration

CAR_FRONT_M = 3.856
CAR_REAR_M  = 1.045
CAR_HALF_W  = 1.09


def make_config():
    cfg = ScenarioConfiguration()
    cfg.name = "Parking"; cfg.type = "Parking"
    cfg.town = "Town04_Opt"; cfg.other_actors = None; cfg.route = False
    return cfg


def cleanup(scenario, world):
    if scenario:
        try: scenario.cleanup()
        except Exception as e: print(f"cleanup: {e}")
    for _ in range(3): world.tick()


class _WP:
    def __init__(self, x, y, angle=0.0):
        self.x = x; self.y = y; self.angle = angle


# ── verbatim from runner_test_medium ─────────────────────────────────────────
def obb_aabb_overlap(cur, front_m, rear_m, half_w, aabb):
    wx  = (aabb[0] + aabb[2]) / 2
    wy  = (aabb[1] + aabb[3]) / 2
    wdx = (aabb[2] - aabb[0]) / 2
    wdy = (aabb[3] - aabb[1]) / 2
    cos_a, sin_a = np.cos(cur.angle), np.sin(cur.angle)
    half_len = (front_m + rear_m) / 2
    obb_cx = cur.x + (front_m - rear_m) / 2 * cos_a
    obb_cy = cur.y + (front_m - rear_m) / 2 * sin_a
    dx, dy = wx - obb_cx, wy - obb_cy
    for ax, ay in [(1, 0), (0, 1), (cos_a, sin_a), (-sin_a, cos_a)]:
        obb_proj  = abs(half_len * (ax*cos_a + ay*sin_a)) + abs(half_w * (-ax*sin_a + ay*cos_a))
        aabb_proj = abs(wdx * ax) + abs(wdy * ay)
        if abs(dx*ax + dy*ay) > obb_proj + aabb_proj:
            return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Scenario-spawned pedestrian visible to get_bounding_boxes()
# ═══════════════════════════════════════════════════════════════════════════════

def test_scenario_pedestrian_visible_in_get_bounding_boxes(carla_world):
    """
    ParkingScenarioHard (PedMode) internally creates a PedestrianCrossingParking
    scenario which spawns walkers via CarlaDataProvider.request_new_actor.

    get_bounding_boxes() queries world.get_actors().filter('walker.*').
    The scenario-spawned walkers must appear there — if they are invisible,
    ALL pedestrian collisions in the benchmark would be silently missed.
    """
    from parking_scenarios.parking_scenario_hard import ParkingScenarioHard, HardMode
    scenario = None
    world    = carla_world
    try:
        scenario = ParkingScenarioHard(
            world=world, config=make_config(),
            destination=35, parked=[34, 36],
            criteria_enable=False, mode=HardMode.PedMode,
        )
        for _ in range(5): world.tick()

        # Verify the pedestrian sub-scenario actually spawned walkers
        ped_scenario = scenario.list_scenarios[0]
        assert len(ped_scenario.other_actors) > 0, \
            "PedestrianCrossingParking spawned no walkers"

        scenario_walker_ids = {a.id for a in ped_scenario.other_actors
                               if 'walker' in a.type_id and 'controller' not in a.type_id}
        assert len(scenario_walker_ids) > 0, \
            "No walker actors in PedestrianCrossingParking.other_actors"

        # Now check get_bounding_boxes() sees them
        _, _, walker_bbs = get_bounding_boxes(scenario)
        visible_walker_ids = {wid for wid, _ in walker_bbs}

        missing = scenario_walker_ids - visible_walker_ids
        assert not missing, (
            f"Scenario-spawned walker(s) {missing} are NOT returned by "
            f"get_bounding_boxes().  Production would silently miss their collisions.\n"
            f"Visible walker ids: {visible_walker_ids}"
        )

    finally:
        cleanup(scenario, world)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Opposite vehicle visible in moving_cars_bbs
# ═══════════════════════════════════════════════════════════════════════════════

def test_opposite_vehicle_visible_in_moving_cars_bbs(carla_world):
    """
    ParkingScenarioMedium spawns an OppositeDirectionVehicle.  That vehicle
    must appear in get_bounding_boxes() moving_cars_bbs — not filtered out as
    ego or parked car.  If it's invisible, the collision_tracker built from
    all_bbs_list would miss it and the vehicle could drive through the ego.
    """
    from parking_scenarios.parking_scenario_medium import ParkingScenarioMedium
    from parking_scenarios.opposite_vehicle_parking import CollisionMode
    scenario = None
    world    = carla_world
    try:
        scenario = ParkingScenarioMedium(
            world=world, config=make_config(),
            destination=28, parked=[27, 29],
            criteria_enable=False, mode=CollisionMode.COLLIDE,
        )
        for _ in range(5): world.tick()

        opp_scenario = scenario.list_scenarios[0]
        assert len(opp_scenario.other_actors) > 0, \
            "OppositeDirectionVehicle has no other_actors"

        opp_actor = opp_scenario.other_actors[0]

        moving_bbs, _, _ = get_bounding_boxes(scenario)

        # Each BB in moving_bbs should cover the opposite vehicle's location
        opp_loc = opp_actor.get_location()
        covered = any(
            bb[0] <= opp_loc.x <= bb[2] and bb[1] <= opp_loc.y <= bb[3]
            for bb in moving_bbs
        )
        assert covered, (
            f"Opposite vehicle at ({opp_loc.x:.1f},{opp_loc.y:.1f}) not covered "
            f"by any moving_cars_bbs.  moving_bbs={moving_bbs}"
        )

    finally:
        cleanup(scenario, world)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. collision_tracker boundary covers the ego's approach road
# ═══════════════════════════════════════════════════════════════════════════════

def test_collision_tracker_covers_ego_start(carla_world):
    """
    Production builds:
        collision_tracker = obstacle_map_from_bbs(all_bbs_list)   # NO min_y

    where all_bbs_list = parked_cars_bbs + cone_bbs + moving_bbs + walker_bbs.

    If all actors happen to be far from the ego start (y≈-243.8), the
    collision_tracker's lower y bound would be higher than the ego's position,
    and generate_collision_mask would be all-zeros for the first N ticks
    (silent blind spot on the approach road).

    This test verifies that the collision_tracker built from a real scenario's
    all_bbs_list has obs_min_y low enough that the ego start is INSIDE the map.
    """
    from parking_scenarios.parking_scenario_easy import ParkingScenarioEasy
    scenario = None
    world    = carla_world
    try:
        scenario = ParkingScenarioEasy(
            world=world, config=make_config(),
            destination=22, parked=[21, 23], criteria_enable=False,
        )
        for _ in range(5): world.tick()

        moving_bbs, cone_bbs, walker_bbs = get_bounding_boxes(scenario)
        all_bbs_list = (scenario.parked_cars_bbs + cone_bbs + moving_bbs
                        + [bb for _, bb in walker_bbs])

        assert len(all_bbs_list) > 0, "all_bbs_list is empty — nothing to build tracker from"

        collision_tracker = obstacle_map_from_bbs(all_bbs_list)

        ego_loc = scenario.car.actor.get_location()

        # The map must extend at least to the ego's rear (y - rear_m - 1 cell)
        min_required_y = ego_loc.y - CAR_REAR_M - 0.25
        assert collision_tracker.min_y <= min_required_y, (
            f"collision_tracker.min_y={collision_tracker.min_y:.2f} is ABOVE ego rear "
            f"at y={min_required_y:.2f}.\n"
            f"The approach road has no collision coverage for the first N ticks!\n"
            f"Fix: pass min_y=ego_loc.y-5 to obstacle_map_from_bbs for collision_tracker "
            f"(same as for the static obs)."
        )

    finally:
        cleanup(scenario, world)


def test_collision_mask_is_nonzero_at_ego_start_with_production_tracker(carla_world):
    """
    Place a cone directly in the ego's path and verify the production-built
    collision_tracker can detect it at the ego's starting position.

    This fails if collision_tracker.min_y > ego start y, because
    generate_collision_mask would return all-zeros (blind spot).
    """
    from parking_scenarios.parking_scenario_easy import ParkingScenarioEasy
    scenario = None
    world    = carla_world
    extra    = []
    try:
        scenario = ParkingScenarioEasy(
            world=world, config=make_config(),
            destination=22, parked=[21, 23], criteria_enable=False,
        )
        for _ in range(5): world.tick()

        ego_loc   = scenario.car.actor.get_location()
        ego_angle = np.pi / 2  # +y direction

        # Spawn a cone 2 m ahead of the ego (inside CAR_FRONT_M=3.856)
        cone_bp = world.get_blueprint_library().find('static.prop.constructioncone')
        cone = world.try_spawn_actor(
            cone_bp,
            carla.Transform(carla.Location(x=ego_loc.x, y=ego_loc.y + 2.0, z=0.3))
        )
        assert cone is not None, "Failed to spawn cone ahead of ego start"
        extra.append(cone)
        world.tick(); world.tick()

        # Build all_bbs_list exactly as production does
        moving_bbs, cone_bbs, walker_bbs = get_bounding_boxes(scenario)
        all_bbs_list = (scenario.parked_cars_bbs + cone_bbs + moving_bbs
                        + [bb for _, bb in walker_bbs])
        collision_tracker = obstacle_map_from_bbs(all_bbs_list)

        car_wp = _WP(ego_loc.x, ego_loc.y, ego_angle)
        collision_mask = collision_tracker.generate_collision_mask(
            car_wp, front_m=CAR_FRONT_M, rear_m=CAR_REAR_M, half_width_m=CAR_HALF_W
        )

        # The mask must be non-zero (ego footprint cells are mapped)
        assert np.any(collision_mask), (
            f"collision_mask is all-zeros at ego start ({ego_loc.x:.1f},{ego_loc.y:.1f}). "
            f"collision_tracker.min_y={collision_tracker.min_y:.2f} — map doesn't reach ego. "
            f"Production is blind for the first ticks!"
        )

        # Now check the cone pipeline detects the cone
        _c = obstacle_map_from_bbs(cone_bbs, collision_tracker).obs
        _c[0,:]=0; _c[-1,:]=0; _c[:,0]=0; _c[:,-1]=0
        dilated = _c | np.roll(_c,1,0)|np.roll(_c,-1,0)|np.roll(_c,1,1)|np.roll(_c,-1,1)
        detected = bool(np.any(collision_mask & (dilated == 1)))
        assert detected, (
            f"Cone 2m ahead of ego NOT detected via production collision_tracker. "
            f"min_y={collision_tracker.min_y:.2f}, ego_y={ego_loc.y:.2f}"
        )

    finally:
        for a in extra:
            try:
                if a and a.is_alive: a.destroy()
            except Exception: pass
        cleanup(scenario, world)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Scenario tree ticking + collision detection still works
# ═══════════════════════════════════════════════════════════════════════════════

def test_scenario_tree_tick_does_not_disrupt_collision_detection(carla_world):
    """
    Production calls scenario_tree.tick_once() BEFORE collision detection
    every loop tick.  Verify that ticking the tree for 10 ticks does not
    corrupt actor positions or break get_bounding_boxes() + obb_aabb_overlap.

    A walker is placed inside the ego footprint.  After each tree tick, the
    walker must still be detected.
    """
    from parking_scenarios.parking_scenario_easy import ParkingScenarioEasy
    scenario = None
    world    = carla_world
    extra    = []
    try:
        scenario = ParkingScenarioEasy(
            world=world, config=make_config(),
            destination=22, parked=[21, 23], criteria_enable=False,
        )
        for _ in range(3): world.tick()

        ref_x, ref_y = 285.6, -225.0
        car_wp       = _WP(ref_x, ref_y, np.pi / 2)

        walker_bp = world.get_blueprint_library().find('walker.pedestrian.0001')
        walker = world.try_spawn_actor(
            walker_bp,
            carla.Transform(carla.Location(x=ref_x, y=ref_y + 2.0, z=0.5))
        )
        assert walker is not None
        extra.append(walker)
        world.tick(); world.tick()

        missed_ticks = []
        for tick_i in range(10):
            # Replicate production loop order exactly
            world.tick()
            timestamp = world.get_snapshot().timestamp
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()

            if scenario.scenario_tree:
                scenario.scenario_tree.tick_once()

            _, _, walker_bbs = get_bounding_boxes(scenario)
            ids = {wid for wid, bb in walker_bbs
                   if obb_aabb_overlap(car_wp, CAR_FRONT_M, CAR_REAR_M, CAR_HALF_W, bb)}

            if walker.id not in ids:
                missed_ticks.append(tick_i)

        assert not missed_ticks, (
            f"Walker in footprint NOT detected on tick(s) {missed_ticks} after "
            f"scenario_tree.tick_once() — tree tick is corrupting detection."
        )

    finally:
        for a in extra:
            try:
                if a and a.is_alive: a.destroy()
            except Exception: pass
        cleanup(scenario, world)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. HardMode.DoorMode scenario spawns correctly
# ═══════════════════════════════════════════════════════════════════════════════

def test_door_mode_scenario_actors_all_alive(carla_world):
    """
    ParkingScenarioHard with HardMode.DoorMode spawns parked cars that have
    dynamic doors.  All other_actors must be alive after construction.

    Only PedMode was covered by the earlier test_scenario_actors_all_alive;
    DoorMode was never tested at all.
    """
    from parking_scenarios.parking_scenario_hard import ParkingScenarioHard, HardMode
    scenario = None
    world    = carla_world
    try:
        scenario = ParkingScenarioHard(
            world=world, config=make_config(),
            destination=35, parked=[34, 36],
            criteria_enable=False, mode=HardMode.DoorMode,
        )
        for _ in range(5): world.tick()

        assert len(scenario.parked_cars) > 0, "DoorMode: no parked cars spawned"

        for actor in scenario.other_actors:
            assert actor is not None, "None in other_actors"
            assert actor.is_alive, (
                f"DoorMode actor {actor.id} ({actor.type_id}) is not alive after spawn"
            )

    finally:
        cleanup(scenario, world)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Scenario pedestrian position updates are tracked by get_bounding_boxes()
# ═══════════════════════════════════════════════════════════════════════════════

def test_moving_pedestrian_position_tracked_by_get_bounding_boxes(carla_world):
    """
    PedestrianCrossingParking moves walkers via set_location() each tick
    (MoveWalkerNoPhysics behavior).  After the behavior tree runs the crossing
    sequence, the walker's position must change AND get_bounding_boxes() must
    report the updated BB each tick.

    We trigger movement by teleporting the ego inside the min_trigger_dist and
    ticking the behavior tree to activate the crossing sequence.
    """
    from parking_scenarios.parking_scenario_hard import ParkingScenarioHard, HardMode
    scenario = None
    world    = carla_world
    try:
        scenario = ParkingScenarioHard(
            world=world, config=make_config(),
            destination=35, parked=[34, 36],
            criteria_enable=False, mode=HardMode.PedMode,
        )
        for _ in range(5): world.tick()

        ped_scenario  = scenario.list_scenarios[0]
        walker_actors = [a for a in ped_scenario.other_actors
                         if 'walker' in a.type_id and 'controller' not in a.type_id]
        assert len(walker_actors) > 0

        walker = walker_actors[0]

        # Record initial walker position
        initial_loc = walker.get_location()

        # Move the ego close to the trigger location (within min_trigger_dist=16.5m)
        # so the InTriggerDistanceToLocation condition fires and the walker starts moving
        dest_loc = parking_vehicle_locations_Town04[35]
        close_loc = carla.Location(x=dest_loc.x, y=dest_loc.y - 10.0, z=0.3)
        scenario.car.actor.set_transform(
            carla.Transform(close_loc, carla.Rotation(yaw=90.0))
        )
        world.tick(); world.tick()

        # Tick scenario tree and collect walker positions from get_bounding_boxes()
        positions = []
        for _ in range(40):
            world.tick()
            timestamp = world.get_snapshot().timestamp
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()
            if scenario.scenario_tree:
                scenario.scenario_tree.tick_once()

            _, _, walker_bbs = get_bounding_boxes(scenario)
            walker_bb = next((bb for wid, bb in walker_bbs if wid == walker.id), None)
            if walker_bb is not None:
                positions.append(((walker_bb[0]+walker_bb[2])/2,
                                   (walker_bb[1]+walker_bb[3])/2))

        assert len(positions) > 0, (
            "get_bounding_boxes() never returned the scenario walker during 40 ticks"
        )

        # At least one position must differ from the initial one — the walker moved
        moved = any(
            abs(py - initial_loc.y) > 0.1 or abs(px - initial_loc.x) > 0.1
            for px, py in positions
        )
        assert moved, (
            f"Walker position never changed over 40 behavior-tree ticks.\n"
            f"Initial: ({initial_loc.x:.2f},{initial_loc.y:.2f})\n"
            f"Observed positions: {positions[:5]}..."
        )

    finally:
        cleanup(scenario, world)
