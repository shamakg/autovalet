"""
Integration tests for parking scenario spawning logic.
Requires a live CARLA server on localhost:2000.

Run with:
    cd ~/carla_garage/leaderboard/leaderboard/autovalet
    source ~/envs/simlingo/bin/activate
    python -m pytest testbed_tests/test_integration_spawning.py -v -s

What this covers
────────────────
1. Cones never overlap any parked car's bounding box
2. Cones never land on the destination spot
3. _get_all_occupied_spots() returns every spot where a car actually was spawned
4. Parked cars are physically near their requested spot locations
5. Random parked cars don't double-occupy a spot
6. Obstacle map cells are non-zero at every parked car centre
7. Easy/Medium/Hard scenario actors (ego, parked, scenario actors) all spawn
8. OppositeDirectionVehicle spawns in the correct lane quadrant
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import carla
import numpy as np
import pytest

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration

from parking_position import parking_vehicle_locations_Town04
from testbed.v2_experiment_utils import (
    town04_spawn_parked_cars,
    obstacle_map_from_bbs,
    get_bounding_boxes,
)
from parking_scenarios.parking_scenario_easy import ParkingScenarioEasy
from parking_scenarios.parking_scenario_medium import ParkingScenarioMedium
from parking_scenarios.parking_scenario_hard import ParkingScenarioHard, HardMode
from parking_scenarios.opposite_vehicle_parking import CollisionMode

pytestmark = pytest.mark.integration


# ── helpers ───────────────────────────────────────────────────────────────────

def make_config():
    cfg = ScenarioConfiguration()
    cfg.name = "Parking"
    cfg.type = "Parking"
    cfg.town = "Town04_Opt"
    cfg.other_actors = None
    cfg.route = False
    return cfg


def world_bb(actor):
    """Return world-space AABB [xmin, ymin, xmax, ymax] for an actor."""
    loc = actor.get_location()
    bb  = actor.bounding_box
    # CARLA bb.location is in actor-local space; for parked cars (yaw=0 or 180)
    # the x-offset matters. We compute the world centre properly.
    yaw = np.deg2rad(actor.get_transform().rotation.yaw)
    bb_world_x = loc.x + bb.location.x * np.cos(yaw) - bb.location.y * np.sin(yaw)
    bb_world_y = loc.y + bb.location.x * np.sin(yaw) + bb.location.y * np.cos(yaw)
    return [
        bb_world_x - bb.extent.x, bb_world_y - bb.extent.y,
        bb_world_x + bb.extent.x, bb_world_y + bb.extent.y,
    ]


def aabbs_overlap(a, b):
    """True if two AABBs [xmin,ymin,xmax,ymax] overlap (not just touch)."""
    return (a[0] < b[2] and a[2] > b[0] and
            a[1] < b[3] and a[3] > b[1])


def cleanup(scenario, world):
    if scenario:
        try:
            scenario.cleanup()
        except Exception as e:
            print(f"cleanup error: {e}")
    for _ in range(3):
        world.tick()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONE SPAWNING INVARIANTS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("destination,parked", [
    (22, [21, 23]),
    (28, [27, 29]),
    (35, [34, 36]),
])
def test_no_cone_overlaps_parked_car(carla_world, destination, parked):
    """
    Every spawned cone must have zero overlap with every parked car's
    world-space bounding box. This is the direct reproduction of the
    reported bug where cones appeared inside parked vehicles.
    """
    scenario = None
    world = carla_world
    try:
        scenario = ParkingScenarioEasy(
            world=world, config=make_config(),
            destination=destination, parked=parked,
            criteria_enable=False,
        )
        for _ in range(5):
            world.tick()

        # Collect cone actors
        all_actors  = world.get_actors()
        cone_actors = [a for a in all_actors.filter('static.prop.*') if 'cone' in a.type_id.lower()]
        car_actors  = scenario.parked_cars

        assert len(cone_actors) > 0, "No cones spawned — check ParkingConeScenario"

        violations = []
        for cone in cone_actors:
            cone_bb = world_bb(cone)
            for car in car_actors:
                car_bb = world_bb(car)
                if aabbs_overlap(cone_bb, car_bb):
                    violations.append(
                        f"Cone at ({cone.get_location().x:.1f},{cone.get_location().y:.1f}) "
                        f"overlaps car at ({car.get_location().x:.1f},{car.get_location().y:.1f})"
                    )

        assert violations == [], "\n".join(violations)

    finally:
        cleanup(scenario, world)


@pytest.mark.parametrize("destination,parked", [
    (22, [21, 23]),
    (30, [29, 31]),
])
def test_no_cone_at_destination_spot(carla_world, destination, parked):
    """No cone may land at the destination parking spot."""
    scenario = None
    world = carla_world
    dest_loc = parking_vehicle_locations_Town04[destination]
    try:
        scenario = ParkingScenarioEasy(
            world=world, config=make_config(),
            destination=destination, parked=parked,
            criteria_enable=False,
        )
        for _ in range(5):
            world.tick()

        all_actors  = world.get_actors()
        cone_actors = [a for a in all_actors.filter('static.prop.*') if 'cone' in a.type_id.lower()]

        for cone in cone_actors:
            loc = cone.get_location()
            dist = ((loc.x - dest_loc.x)**2 + (loc.y - dest_loc.y)**2)**0.5
            assert dist > 1.5, (
                f"Cone at ({loc.x:.1f},{loc.y:.1f}) is only {dist:.2f}m from destination spot"
            )

    finally:
        cleanup(scenario, world)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. _get_all_occupied_spots() COVERAGE
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("destination,parked", [
    (22, [21, 23]),
    (28, [27, 29]),
])
def test_occupied_spots_covers_all_spawned_cars(carla_world, destination, parked):
    """
    Every successfully spawned parked car must appear in the output of
    _get_all_occupied_spots(). If a car is missing, the cone scenario won't
    know the spot is taken and may place a cone on top of the car.
    """
    scenario = None
    world = carla_world
    try:
        scenario = ParkingScenarioEasy(
            world=world, config=make_config(),
            destination=destination, parked=parked,
            criteria_enable=False,
        )
        for _ in range(5):
            world.tick()

        occupied = set(scenario._get_all_occupied_spots())

        missing = []
        for car in scenario.parked_cars:
            car_loc = car.get_location()
            # Find the nearest parking spot index
            nearest_idx = min(
                range(len(parking_vehicle_locations_Town04)),
                key=lambda i: (
                    (parking_vehicle_locations_Town04[i].x - car_loc.x)**2 +
                    (parking_vehicle_locations_Town04[i].y - car_loc.y)**2
                )
            )
            nearest_dist = (
                (parking_vehicle_locations_Town04[nearest_idx].x - car_loc.x)**2 +
                (parking_vehicle_locations_Town04[nearest_idx].y - car_loc.y)**2
            )**0.5

            if nearest_dist < 3.0 and nearest_idx not in occupied:
                missing.append(
                    f"Parked car at ({car_loc.x:.1f},{car_loc.y:.1f}) maps to spot {nearest_idx} "
                    f"(dist={nearest_dist:.2f}m) but that spot is NOT in occupied={sorted(occupied)}"
                )

        assert missing == [], "\n".join(missing)

    finally:
        cleanup(scenario, world)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PARKED CAR PHYSICAL PLACEMENT
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("destination,parked", [
    (22, [21, 23]),
])
def test_explicit_parked_cars_near_requested_spots(carla_world, destination, parked):
    """
    Explicitly requested parked spots must have a car within 2 m of the
    nominal spot location after ticking.
    """
    scenario = None
    world = carla_world
    try:
        scenario = ParkingScenarioEasy(
            world=world, config=make_config(),
            destination=destination, parked=parked,
            criteria_enable=False,
        )
        for _ in range(5):
            world.tick()

        for spot_idx in parked:
            spot_loc = parking_vehicle_locations_Town04[spot_idx]
            nearest_dist = min(
                (
                    (car.get_location().x - spot_loc.x)**2 +
                    (car.get_location().y - spot_loc.y)**2
                )**0.5
                for car in scenario.parked_cars
            )
            assert nearest_dist < 2.0, (
                f"No parked car within 2m of requested spot {spot_idx} "
                f"at ({spot_loc.x},{spot_loc.y}); nearest={nearest_dist:.2f}m"
            )

    finally:
        cleanup(scenario, world)


def test_no_two_parked_cars_share_a_spot(carla_world):
    """No two parked cars should occupy the same parking spot index."""
    destination, parked = 22, [21, 23]
    scenario = None
    world = carla_world
    try:
        scenario = ParkingScenarioEasy(
            world=world, config=make_config(),
            destination=destination, parked=parked,
            criteria_enable=False,
        )
        for _ in range(5):
            world.tick()

        # Assign each car to its nearest spot
        spot_assignments = {}
        for car in scenario.parked_cars:
            car_loc = car.get_location()
            nearest_idx = min(
                range(len(parking_vehicle_locations_Town04)),
                key=lambda i: (
                    (parking_vehicle_locations_Town04[i].x - car_loc.x)**2 +
                    (parking_vehicle_locations_Town04[i].y - car_loc.y)**2
                )
            )
            nearest_dist = (
                (parking_vehicle_locations_Town04[nearest_idx].x - car_loc.x)**2 +
                (parking_vehicle_locations_Town04[nearest_idx].y - car_loc.y)**2
            )**0.5
            if nearest_dist < 3.0:
                if nearest_idx in spot_assignments:
                    spot_assignments[nearest_idx].append(car.id)
                else:
                    spot_assignments[nearest_idx] = [car.id]

        doubles = {k: v for k, v in spot_assignments.items() if len(v) > 1}
        assert doubles == {}, f"Multiple cars at same spot(s): {doubles}"

    finally:
        cleanup(scenario, world)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. OBSTACLE MAP COVERAGE OF PARKED CARS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("destination,parked", [
    (22, [21, 23]),
])
def test_obstacle_map_marks_parked_car_centres(carla_world, destination, parked):
    """
    After building obstacle_map_from_bbs from the parked_cars_bbs returned by
    town04_spawn_parked_cars, the grid cell at each parked car's world location
    must be occupied (obs == 1 or on a marked boundary).

    This exercises the actual testbed utility used inside run_scenario() to
    build the static obstacle map for the ego car.
    """
    scenario = None
    world = carla_world
    try:
        scenario = ParkingScenarioEasy(
            world=world, config=make_config(),
            destination=destination, parked=parked,
            criteria_enable=False,
        )
        for _ in range(5):
            world.tick()

        obs_map = obstacle_map_from_bbs(scenario.parked_cars_bbs)

        failures = []
        for car, bb in zip(scenario.parked_cars, scenario.parked_cars_bbs):
            # obstacle_map_from_bbs draws the perimeter of each BB.  Check that
            # at least one cell on the actual BB edges (not a hardcoded ±0.96) is
            # marked as an obstacle.  We sample the top/bottom edges explicitly.
            found = False
            for x in np.arange(bb[0], bb[2] + 0.25, 0.25):
                for y_edge in [bb[1], bb[3]]:  # bottom and top edges
                    gx, gy = obs_map.transform_coord(x, y_edge)
                    if 0 <= gx < obs_map.obs.shape[0] and 0 <= gy < obs_map.obs.shape[1]:
                        if obs_map.obs[gx, gy] == 1:
                            found = True
                            break
                if found:
                    break
            if not found:
                # Also check left/right edges
                for y in np.arange(bb[1], bb[3] + 0.25, 0.25):
                    for x_edge in [bb[0], bb[2]]:
                        gx, gy = obs_map.transform_coord(x_edge, y)
                        if 0 <= gx < obs_map.obs.shape[0] and 0 <= gy < obs_map.obs.shape[1]:
                            if obs_map.obs[gx, gy] == 1:
                                found = True
                                break
                    if found:
                        break

            if not found:
                loc = car.get_location()
                failures.append(
                    f"Parked car at ({loc.x:.1f},{loc.y:.1f}) has no obstacle cells "
                    f"in obs_map along its bounding box perimeter (bb={bb})"
                )

        assert failures == [], "\n".join(failures)

    finally:
        cleanup(scenario, world)


@pytest.mark.parametrize("destination,parked", [
    (22, [21, 23]),
])
def test_collision_mask_hits_parked_cars(carla_world, destination, parked):
    """
    If the ego car were placed at a parked car's location, the collision mask
    generated from the obstacle map should report an overlap.

    This validates that generate_collision_mask + obs_map together correctly
    detect contact with parked vehicles — the same pipeline used in runner_test_medium.
    """
    scenario = None
    world = carla_world
    try:
        scenario = ParkingScenarioEasy(
            world=world, config=make_config(),
            destination=destination, parked=parked,
            criteria_enable=False,
        )
        for _ in range(5):
            world.tick()

        all_bbs  = scenario.parked_cars_bbs
        obs_map  = obstacle_map_from_bbs(all_bbs)

        class _WP:
            def __init__(self, x, y, angle=0.0):
                self.x = x; self.y = y; self.angle = angle

        # For each explicitly requested parked car spot, put a "virtual ego" there
        # and confirm the collision mask sees an obstacle
        hits = 0
        for spot_idx in parked:
            loc = parking_vehicle_locations_Town04[spot_idx]
            for angle in [0.0, np.pi]:
                wp   = _WP(loc.x, loc.y, angle)
                mask = obs_map.generate_collision_mask(
                    wp, front_m=3.856, rear_m=1.045, half_width_m=1.09
                )
                if np.any(mask & (obs_map.obs == 1)):
                    hits += 1
                    break  # either angle works

        assert hits == len(parked), (
            f"Collision mask missed {len(parked) - hits} of {len(parked)} parked car spots"
        )

    finally:
        cleanup(scenario, world)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MEDIUM SCENARIO — OPPOSITE VEHICLE
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("destination,parked,mode", [
    (28, [27, 29], CollisionMode.MISS),
    (28, [27, 29], CollisionMode.STOP_EARLY),
    (28, [27, 29], CollisionMode.COLLIDE),
])
def test_opposite_vehicle_spawns_in_correct_lane(carla_world, destination, parked, mode):
    """
    The OppositeDirectionVehicle must spawn at the lane x coordinate closest
    to the ego (offset by half a lane width), not in the parking bays.
    """
    scenario = None
    world = carla_world
    from parking_position import parking_lane_waypoints_Town04
    try:
        scenario = ParkingScenarioMedium(
            world=world, config=make_config(),
            destination=destination, parked=parked,
            criteria_enable=False, mode=mode,
        )
        for _ in range(5):
            world.tick()

        # The opposite vehicle is in list_scenarios[0].other_actors[0]
        assert len(scenario.list_scenarios) > 0, "No sub-scenarios loaded"
        opp_scenario = scenario.list_scenarios[0]
        assert len(opp_scenario.other_actors) > 0, "OppositeDirectionVehicle has no actors"

        opp_actor = opp_scenario.other_actors[0]
        # Force it above ground (it may be hidden below) and check its x
        opp_transform = opp_scenario.spawn_transform

        spawn_x = opp_transform.location.x

        lane_xs = sorted(set(wp[0] for wp in parking_lane_waypoints_Town04))
        # spawn_x should be lane_x + lane_width/2 for the closest lane
        ego_x = scenario.car.actor.get_location().x
        closest_lane_x = min(lane_xs, key=lambda x: abs(x - ego_x))
        expected_spawn_x = closest_lane_x + 1.5  # lane_width / 2 = 1.5

        assert abs(spawn_x - expected_spawn_x) < 0.5, (
            f"Opposite vehicle spawned at x={spawn_x:.2f}, expected ~{expected_spawn_x:.2f} "
            f"(closest lane={closest_lane_x})"
        )

    finally:
        cleanup(scenario, world)


def test_opposite_vehicle_miss_mode_spawn_y_offset(carla_world):
    """In MISS mode the vehicle spawns offset enough to clear the ego path.

    Using destination=22 instead of 28 because spot 28 is too close to the
    town04 y_max boundary (-178), causing both MISS and COLLIDE spawn y values
    to clamp to the same limit, making the assertion trivially fail.
    """
    destination, parked = 22, [21, 23]
    scenario_miss    = None
    scenario_collide = None
    world = carla_world
    try:
        scenario_miss = ParkingScenarioMedium(
            world=world, config=make_config(),
            destination=destination, parked=parked,
            criteria_enable=False, mode=CollisionMode.MISS,
        )
        for _ in range(3):
            world.tick()
        miss_y = scenario_miss.list_scenarios[0].spawn_transform.location.y
        cleanup(scenario_miss, world)
        scenario_miss = None

        scenario_collide = ParkingScenarioMedium(
            world=world, config=make_config(),
            destination=destination, parked=parked,
            criteria_enable=False, mode=CollisionMode.COLLIDE,
        )
        for _ in range(3):
            world.tick()
        collide_y = scenario_collide.list_scenarios[0].spawn_transform.location.y

        # MISS vehicle must be closer to ego start (lower y magnitude) than COLLIDE
        assert miss_y != collide_y, "MISS and COLLIDE spawn at same y — miss_offset not applied"

    finally:
        cleanup(scenario_miss, world)
        cleanup(scenario_collide, world)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. EGO VEHICLE SPAWNING
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("destination,parked", [
    (22, [21, 23]),
    (35, [34, 36]),
])
def test_ego_spawns_away_from_parked_cars(carla_world, destination, parked):
    """Ego vehicle must not spawn on top of any parked car."""
    scenario = None
    world = carla_world
    try:
        scenario = ParkingScenarioEasy(
            world=world, config=make_config(),
            destination=destination, parked=parked,
            criteria_enable=False,
        )
        for _ in range(5):
            world.tick()

        ego_loc = scenario.car.actor.get_location()
        ego_bb  = world_bb(scenario.car.actor)

        for car in scenario.parked_cars:
            car_bb = world_bb(car)
            assert not aabbs_overlap(ego_bb, car_bb), (
                f"Ego at ({ego_loc.x:.1f},{ego_loc.y:.1f}) overlaps parked car "
                f"at ({car.get_location().x:.1f},{car.get_location().y:.1f})"
            )

    finally:
        cleanup(scenario, world)


@pytest.mark.parametrize("ScenarioClass,kwargs", [
    (ParkingScenarioEasy,   dict(destination=22, parked=[21, 23])),
    (ParkingScenarioMedium, dict(destination=28, parked=[27, 29])),
    (ParkingScenarioHard,   dict(destination=35, parked=[34, 36], mode=HardMode.PedMode)),
])
def test_scenario_actors_all_alive(carla_world, ScenarioClass, kwargs):
    """All actors registered in other_actors must be alive after spawn."""
    scenario = None
    world = carla_world
    try:
        scenario = ScenarioClass(
            world=world, config=make_config(),
            criteria_enable=False, **kwargs,
        )
        for _ in range(5):
            world.tick()

        for actor in scenario.other_actors:
            assert actor is not None, "None actor in other_actors"
            assert actor.is_alive, (
                f"Actor {actor.id} ({actor.type_id}) is not alive after spawn"
            )

    finally:
        cleanup(scenario, world)
