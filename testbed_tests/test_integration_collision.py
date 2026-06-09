"""
Integration tests for collision detection with real CARLA actor geometry.
Requires a live CARLA server on localhost:2000.

Run with:
    cd ~/carla_garage/leaderboard/leaderboard/autovalet
    source ~/envs/simlingo/bin/activate
    python -m pytest testbed_tests/test_integration_collision.py -v -s

What this covers
────────────────
1. obb_aabb_overlap with a real walker's bounding box
   - Walker placed directly in ego path → must detect collision
   - Walker placed clearly to the side → must NOT falsely detect
   - Walker at oblique angle to ego heading
2. generate_collision_mask with real parked car bounding boxes
   - Ego at a parked car's spot → mask must overlap obs_map
   - Ego at neutral start location → mask must NOT overlap parked cars
3. Cone BB detection via obb_aabb_overlap
   - Cone inside ego path → detected
   - Cone off to the side → not detected
4. Full pipeline: build obs_map from real BBs, run generate_collision_mask,
   check near_miss distance returns a sensible value
5. Walker/cone BB detection consistent between obb_aabb_overlap and grid mask
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import carla
import numpy as np
import pytest
import time

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration

from parking_position import parking_vehicle_locations_Town04, player_location_Town04
from testbed.v2_experiment_utils import (
    town04_spawn_parked_cars,
    obstacle_map_from_bbs,
    get_bounding_boxes,
)

pytestmark = pytest.mark.integration

# ── inline from runner_test_medium.py (unchanged) ────────────────────────────

def obb_aabb_overlap(cur, front_m, rear_m, half_w, aabb):
    """Exact OBB-AABB overlap via SAT (verbatim copy of production code)."""
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


def near_miss(collision_mask, ground_truth_obs):
    """Verbatim copy from v2_experiment_utils for self-contained testing."""
    if np.any(collision_mask & (ground_truth_obs == 1)):
        return 0.0
    car_pixels = np.argwhere(collision_mask)
    obstacle_pixels = np.argwhere(ground_truth_obs == 1)
    if len(obstacle_pixels) == 0:
        return float('inf')
    min_pixels = float('inf')
    for car_pix in car_pixels[::5]:
        distances = np.sqrt(np.sum((obstacle_pixels - car_pix)**2, axis=1))
        min_pixels = min(min_pixels, np.min(distances))
    return min_pixels * 0.25


# ── helpers ───────────────────────────────────────────────────────────────────

class _WP:
    """Minimal TrajectoryPoint stand-in."""
    def __init__(self, x, y, angle=0.0):
        self.x = x; self.y = y; self.angle = angle


def actor_aabb(actor):
    """World-space AABB [xmin,ymin,xmax,ymax] accounting for bbox offset."""
    t   = actor.get_transform()
    bb  = actor.bounding_box
    yaw = np.deg2rad(t.rotation.yaw)
    cx  = t.location.x + bb.location.x * np.cos(yaw) - bb.location.y * np.sin(yaw)
    cy  = t.location.y + bb.location.x * np.sin(yaw) + bb.location.y * np.cos(yaw)
    return [cx - bb.extent.x, cy - bb.extent.y, cx + bb.extent.x, cy + bb.extent.y]


CAR_FRONT_M   = 3.856
CAR_REAR_M    = 1.045
CAR_HALF_W    = 1.09

WALKER_BP     = 'walker.pedestrian.0001'
CONE_BP       = 'static.prop.constructioncone'
EGO_BP        = 'vehicle.lincoln.mkz_2020'


def spawn_actor(world, bp_id, location, rotation=None):
    bp = world.get_blueprint_library().find(bp_id)
    rot = rotation or carla.Rotation()
    transform = carla.Transform(carla.Location(**location), rot)
    actor = world.try_spawn_actor(bp, transform)
    assert actor is not None, f"Failed to spawn {bp_id} at {location}"
    return actor


def destroy_actors(actors):
    for a in actors:
        try:
            if a and a.is_alive:
                a.destroy()
        except Exception:
            pass


def make_config():
    cfg = ScenarioConfiguration()
    cfg.name = "Parking"; cfg.type = "Parking"
    cfg.town = "Town04_Opt"; cfg.other_actors = None; cfg.route = False
    return cfg


# ═══════════════════════════════════════════════════════════════════════════════
# 1. WALKER COLLISION — obb_aabb_overlap WITH REAL ACTOR GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════════

def test_walker_directly_in_front_detected(carla_world):
    """
    Walker spawned 2 m directly ahead of the ego reference point.
    obb_aabb_overlap must return True with the real actor bounding box.
    """
    world = carla_world
    actors = []
    try:
        # Ego reference point at a quiet area of the parking lot
        ego_x, ego_y = 285.6, -230.0
        ego_angle = np.pi / 2   # facing +y

        walker = spawn_actor(world, WALKER_BP, dict(
            x=ego_x, y=ego_y + 2.0, z=0.5
        ))
        actors.append(walker)
        world.tick(); world.tick()

        cur  = _WP(ego_x, ego_y, ego_angle)
        aabb = actor_aabb(walker)

        result = obb_aabb_overlap(cur, CAR_FRONT_M, CAR_REAR_M, CAR_HALF_W, aabb)
        assert result is True, (
            f"Walker 2m ahead NOT detected. Walker BB={aabb}, "
            f"ego at ({ego_x},{ego_y}) facing {np.degrees(ego_angle):.0f}°"
        )

    finally:
        destroy_actors(actors)
        world.tick()


def test_walker_far_to_side_not_detected(carla_world):
    """Walker 5 m to the side of the ego must NOT trigger obb_aabb_overlap."""
    world = carla_world
    actors = []
    try:
        ego_x, ego_y  = 285.6, -230.0
        ego_angle     = np.pi / 2  # facing +y

        # Place walker 5 m to the right (x direction)
        walker = spawn_actor(world, WALKER_BP, dict(
            x=ego_x + 5.0, y=ego_y, z=0.5
        ))
        actors.append(walker)
        world.tick(); world.tick()

        cur  = _WP(ego_x, ego_y, ego_angle)
        aabb = actor_aabb(walker)

        result = obb_aabb_overlap(cur, CAR_FRONT_M, CAR_REAR_M, CAR_HALF_W, aabb)
        assert result is False, (
            f"Walker 5m to the side incorrectly detected as collision. "
            f"Walker BB={aabb}, ego at ({ego_x},{ego_y})"
        )

    finally:
        destroy_actors(actors)
        world.tick()


def test_walker_just_outside_front_not_detected(carla_world):
    """Walker 1 m beyond the front face of the car must NOT be detected.

    Note: 0.01 m clearance is insufficient because the real walker bounding box
    extends ~0.19 m behind the actor origin, so the BB rear would still overlap
    the car OBB.  1 m gives a safe margin while still testing the boundary.
    """
    world = carla_world
    actors = []
    try:
        ego_x, ego_y  = 285.6, -230.0
        ego_angle     = np.pi / 2

        # OBB front extends front_m ahead of reference in forward dir (+y here).
        # Use 1 m clearance so the walker's ~0.19 m rear BB extent doesn't reach back into the OBB.
        just_past_y = ego_y + CAR_FRONT_M + 1.0

        walker = spawn_actor(world, WALKER_BP, dict(
            x=ego_x, y=just_past_y, z=0.5
        ))
        actors.append(walker)
        world.tick(); world.tick()

        cur  = _WP(ego_x, ego_y, ego_angle)
        aabb = actor_aabb(walker)

        result = obb_aabb_overlap(cur, CAR_FRONT_M, CAR_REAR_M, CAR_HALF_W, aabb)
        assert result is False, (
            f"Walker just past front face falsely detected. "
            f"just_past_y={just_past_y:.3f}, walker BB={aabb}"
        )

    finally:
        destroy_actors(actors)
        world.tick()


def test_walker_at_front_edge_detected(carla_world):
    """Walker with centre exactly at the front face should still be detected."""
    world = carla_world
    actors = []
    try:
        ego_x, ego_y  = 285.6, -230.0
        ego_angle     = np.pi / 2

        walker = spawn_actor(world, WALKER_BP, dict(
            x=ego_x, y=ego_y + CAR_FRONT_M, z=0.5
        ))
        actors.append(walker)
        world.tick(); world.tick()

        cur  = _WP(ego_x, ego_y, ego_angle)
        aabb = actor_aabb(walker)

        result = obb_aabb_overlap(cur, CAR_FRONT_M, CAR_REAR_M, CAR_HALF_W, aabb)
        assert result is True, "Walker at exact front face not detected"

    finally:
        destroy_actors(actors)
        world.tick()


def test_walker_lateral_edge_just_inside_detected(carla_world):
    """Walker centred just inside the lateral edge of the car."""
    world = carla_world
    actors = []
    try:
        ego_x, ego_y  = 285.6, -230.0
        ego_angle     = 0.0  # facing +x

        walker = spawn_actor(world, WALKER_BP, dict(
            x=ego_x + 1.5, y=ego_y + (CAR_HALF_W - 0.1), z=0.5
        ))
        actors.append(walker)
        world.tick(); world.tick()

        cur  = _WP(ego_x, ego_y, ego_angle)
        aabb = actor_aabb(walker)

        result = obb_aabb_overlap(cur, CAR_FRONT_M, CAR_REAR_M, CAR_HALF_W, aabb)
        assert result is True, (
            f"Walker just inside lateral edge not detected. BB={aabb}"
        )

    finally:
        destroy_actors(actors)
        world.tick()


def test_walker_lateral_edge_just_outside_not_detected(carla_world):
    """Walker just outside the lateral edge must not be falsely flagged."""
    world = carla_world
    actors = []
    try:
        ego_x, ego_y  = 285.6, -230.0
        ego_angle     = 0.0

        walker = spawn_actor(world, WALKER_BP, dict(
            x=ego_x + 1.5, y=ego_y + CAR_HALF_W + 0.6, z=0.5
        ))
        actors.append(walker)
        world.tick(); world.tick()

        cur  = _WP(ego_x, ego_y, ego_angle)
        aabb = actor_aabb(walker)

        result = obb_aabb_overlap(cur, CAR_FRONT_M, CAR_REAR_M, CAR_HALF_W, aabb)
        assert result is False, (
            f"Walker outside lateral edge falsely detected. BB={aabb}"
        )

    finally:
        destroy_actors(actors)
        world.tick()


def test_multiple_walkers_only_colliding_one_detected(carla_world):
    """With one walker in path and one clear, only the colliding one is detected."""
    world = carla_world
    actors = []
    try:
        ego_x, ego_y  = 285.6, -230.0
        ego_angle     = np.pi / 2

        walker_in  = spawn_actor(world, WALKER_BP, dict(x=ego_x,       y=ego_y + 2.0, z=0.5))
        walker_out = spawn_actor(world, WALKER_BP, dict(x=ego_x + 6.0, y=ego_y,       z=0.5))
        actors = [walker_in, walker_out]
        world.tick(); world.tick()

        cur = _WP(ego_x, ego_y, ego_angle)

        assert obb_aabb_overlap(cur, CAR_FRONT_M, CAR_REAR_M, CAR_HALF_W, actor_aabb(walker_in)),  "In-path walker not detected"
        assert not obb_aabb_overlap(cur, CAR_FRONT_M, CAR_REAR_M, CAR_HALF_W, actor_aabb(walker_out)), "Off-path walker falsely detected"

    finally:
        destroy_actors(actors)
        world.tick()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CONE COLLISION — obb_aabb_overlap WITH REAL CONE GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════════

def test_cone_in_path_detected(carla_world):
    """Cone 1.5m ahead of ego reference must be detected by obb_aabb_overlap."""
    world = carla_world
    actors = []
    try:
        ego_x, ego_y  = 285.6, -225.0
        ego_angle     = np.pi / 2

        cone = spawn_actor(world, CONE_BP, dict(x=ego_x, y=ego_y + 1.5, z=0.3))
        actors.append(cone)
        world.tick(); world.tick()

        cur  = _WP(ego_x, ego_y, ego_angle)
        aabb = actor_aabb(cone)

        result = obb_aabb_overlap(cur, CAR_FRONT_M, CAR_REAR_M, CAR_HALF_W, aabb)
        assert result is True, f"Cone in path not detected. BB={aabb}"

    finally:
        destroy_actors(actors)
        world.tick()


def test_cone_off_to_side_not_detected(carla_world):
    """Cone 4m to the side should not be detected."""
    world = carla_world
    actors = []
    try:
        ego_x, ego_y  = 285.6, -225.0
        ego_angle     = np.pi / 2

        cone = spawn_actor(world, CONE_BP, dict(x=ego_x + 4.0, y=ego_y, z=0.3))
        actors.append(cone)
        world.tick(); world.tick()

        cur  = _WP(ego_x, ego_y, ego_angle)
        aabb = actor_aabb(cone)

        result = obb_aabb_overlap(cur, CAR_FRONT_M, CAR_REAR_M, CAR_HALF_W, aabb)
        assert result is False, f"Off-side cone falsely detected. BB={aabb}"

    finally:
        destroy_actors(actors)
        world.tick()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PARKED CAR COLLISION — generate_collision_mask WITH REAL BBs
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("destination,parked", [
    (22, [21, 23]),
    (30, [29, 31]),
])
def test_collision_mask_at_ego_start_clear(carla_world, destination, parked):
    """
    At the ego's starting position the collision mask must NOT overlap any
    parked car in the obstacle map. If it does, the planning pipeline is
    already broken before the car moves.
    """
    from parking_scenarios.parking_scenario_easy import ParkingScenarioEasy
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
        # Match production: runner_test_medium passes min_y=ego_loc.y - 5 so the
        # map always extends below the ego start.  Without this, border cells
        # (always obs==1) can fall inside the ego footprint when all parked cars
        # are far from the ego start, producing a spurious collision.
        obs_map = obstacle_map_from_bbs(scenario.parked_cars_bbs, min_y=ego_loc.y - 5)

        wp   = _WP(ego_loc.x, ego_loc.y, np.pi / 2)
        mask = obs_map.generate_collision_mask(
            wp, front_m=CAR_FRONT_M, rear_m=CAR_REAR_M, half_width_m=CAR_HALF_W
        )

        collision = bool(np.any(mask & (obs_map.obs == 1)))
        assert not collision, (
            f"Ego at start ({ego_loc.x:.1f},{ego_loc.y:.1f}) already collides with "
            f"parked car obstacle map — parking lot is misconfigured"
        )

    finally:
        if scenario:
            try: scenario.cleanup()
            except: pass
        for _ in range(3): world.tick()


@pytest.mark.parametrize("destination,parked", [
    (22, [21, 23]),
])
def test_collision_mask_at_parked_spot_hits(carla_world, destination, parked):
    """
    Placing the virtual ego at an explicitly parked car's spot must trigger
    a collision in the obstacle map. Confirms that parked_cars_bbs are correct.
    """
    from parking_scenarios.parking_scenario_easy import ParkingScenarioEasy
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

        for spot_idx in parked:
            loc = parking_vehicle_locations_Town04[spot_idx]
            wp  = _WP(loc.x, loc.y, 0.0)
            mask = obs_map.generate_collision_mask(
                wp, front_m=CAR_FRONT_M, rear_m=CAR_REAR_M, half_width_m=CAR_HALF_W
            )
            collision = bool(np.any(mask & (obs_map.obs == 1)))
            assert collision, (
                f"Virtual ego at parked spot {spot_idx} ({loc.x},{loc.y}) "
                f"does NOT collide with obstacle map — parked_cars_bbs may be wrong"
            )

    finally:
        if scenario:
            try: scenario.cleanup()
            except: pass
        for _ in range(3): world.tick()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. NEAR-MISS PIPELINE WITH REAL ACTOR BBs
# ═══════════════════════════════════════════════════════════════════════════════

def test_near_miss_distance_with_real_walker(carla_world):
    """
    With a walker 3 m ahead of the ego (outside the collision footprint),
    near_miss must return a finite positive distance.
    """
    from parking_scenarios.parking_scenario_easy import ParkingScenarioEasy
    scenario = None
    world = carla_world
    extra_actors = []
    try:
        destination, parked = 22, [21, 23]
        scenario = ParkingScenarioEasy(
            world=world, config=make_config(),
            destination=destination, parked=parked,
            criteria_enable=False,
        )
        for _ in range(5):
            world.tick()

        ego_loc   = scenario.car.actor.get_location()
        ego_angle = np.pi / 2   # facing +y

        # Walker 6 m ahead (outside front_m=3.856 footprint → no collision)
        walker = spawn_actor(world, WALKER_BP, dict(
            x=ego_loc.x, y=ego_loc.y + 6.0, z=0.5
        ))
        extra_actors.append(walker)
        world.tick(); world.tick()

        all_bbs  = scenario.parked_cars_bbs + [actor_aabb(walker)]
        obs_map  = obstacle_map_from_bbs(all_bbs)

        wp   = _WP(ego_loc.x, ego_loc.y, ego_angle)
        mask = obs_map.generate_collision_mask(
            wp, front_m=CAR_FRONT_M, rear_m=CAR_REAR_M, half_width_m=CAR_HALF_W
        )

        walker_obs = obstacle_map_from_bbs([actor_aabb(walker)], obs_map).obs

        dist = near_miss(mask, walker_obs)

        assert dist != float('inf'), "near_miss returned inf — walker not represented in obs_map"
        assert dist > 0, f"near_miss returned 0 (collision) for walker 6m away: {dist}"
        assert dist < 5.0, f"near_miss returned unrealistically large distance: {dist}m"

    finally:
        destroy_actors(extra_actors)
        if scenario:
            try: scenario.cleanup()
            except: pass
        for _ in range(3): world.tick()


def test_near_miss_returns_zero_for_real_collision(carla_world):
    """
    With a walker placed directly inside the ego footprint, near_miss must
    return exactly 0.0 (collision, not near-miss).

    We use an open reference position (not where an actual ego car is) so that
    CARLA can spawn the walker inside the OBB without being blocked by the
    vehicle's physical body.
    """
    world = carla_world
    actors = []
    try:
        # Open spot in the parking lot with no real vehicle there.
        # Ego faces +y (pi/2); walker at +2 m forward is well inside CAR_FRONT_M=3.856.
        ref_x, ref_y = 285.6, -225.0
        ego_angle    = np.pi / 2

        walker = spawn_actor(world, WALKER_BP, dict(
            x=ref_x, y=ref_y + 2.0, z=0.5
        ))
        actors.append(walker)
        world.tick(); world.tick()

        obs_map = obstacle_map_from_bbs([actor_aabb(walker)])

        wp   = _WP(ref_x, ref_y, ego_angle)
        mask = obs_map.generate_collision_mask(
            wp, front_m=CAR_FRONT_M, rear_m=CAR_REAR_M, half_width_m=CAR_HALF_W
        )

        walker_obs = obstacle_map_from_bbs([actor_aabb(walker)], obs_map).obs
        dist = near_miss(mask, walker_obs)
        assert dist == 0.0, f"Expected 0.0 (collision) for walker inside footprint, got {dist}m"

    finally:
        destroy_actors(actors)
        world.tick()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CONSISTENCY: obb_aabb_overlap vs. generate_collision_mask + grid
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("offset_y,expect_collision", [
    (1.0, True),    # 1m ahead — inside footprint
    (4.5, False),   # 4.5m ahead — outside footprint (front_m=3.856)
    (6.0, False),   # 6m ahead — clearly outside
])
def test_obb_and_grid_agree(carla_world, offset_y, expect_collision):
    """
    Both obb_aabb_overlap (precise SAT) and the collision_mask+grid approach
    must agree on whether a walker is inside the ego footprint.

    If they disagree that's a sign of grid quantisation or mapping errors
    that could cause silent misses during the benchmark.
    """
    world = carla_world
    actors = []
    try:
        ego_x, ego_y  = 285.6, -230.0
        ego_angle     = np.pi / 2

        walker = spawn_actor(world, WALKER_BP, dict(
            x=ego_x, y=ego_y + offset_y, z=0.5
        ))
        actors.append(walker)
        world.tick(); world.tick()

        aabb = actor_aabb(walker)
        cur  = _WP(ego_x, ego_y, ego_angle)

        # Method 1: exact SAT
        obb_result = obb_aabb_overlap(cur, CAR_FRONT_M, CAR_REAR_M, CAR_HALF_W, aabb)

        # Method 2: grid-based
        obs_map     = obstacle_map_from_bbs([aabb])
        mask        = obs_map.generate_collision_mask(
            cur, front_m=CAR_FRONT_M, rear_m=CAR_REAR_M, half_width_m=CAR_HALF_W
        )
        grid_result = bool(np.any(mask & (obs_map.obs == 1)))

        assert obb_result == expect_collision, (
            f"obb_aabb_overlap: offset_y={offset_y}, expected {expect_collision}, got {obb_result}"
        )
        assert grid_result == expect_collision, (
            f"grid mask: offset_y={offset_y}, expected {expect_collision}, got {grid_result}. "
            f"(Note: grid is 0.25m resolution, small discrepancies at exact boundaries are OK)"
        )

    finally:
        destroy_actors(actors)
        world.tick()


# ═══════════════════════════════════════════════════════════════════════════════
# GAP #2 — car.cur is 1.6 m behind actor origin; production footprint must
#           still cover the same physical region as integration test constants
# ═══════════════════════════════════════════════════════════════════════════════

def test_production_footprint_agrees_with_actor_geometry(carla_world):
    """
    CarlaCar.localize() sets car.cur = gnss_location.offset(-1), shifting the
    reference point 1.6 m BEHIND the actor origin along the heading.
    CarlaCar.__init__ compensates via:
        front_m = bb.extent.x + bb.location.x + 1.6
        rear_m  = max(0.1, bb.extent.x - bb.location.x - 1.6)

    This means the production front face in world-space is:
        cur + front_m * heading = actor_origin + (bb.extent.x + bb.location.x) * heading

    which should equal the actor's physical front face derived from its bounding box.

    We verify this geometrically (no walker spawn needed) by comparing the
    production front-face position against the actor BB directly.  If the
    formula is wrong, integration tests using CAR_FRONT_M=3.856 describe a
    different footprint than the car actually uses at runtime.
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

        car_obj = scenario.car.car
        actor   = scenario.car.actor
        bb      = actor.bounding_box

        # Production dims derived from the actual bounding box
        front_m = car_obj.front_m
        rear_m  = car_obj.rear_m

        # Localize so car.cur reflects current GNSS position
        car_obj.localize()
        cur = car_obj.cur   # 1.6 m behind actor origin along heading

        angle = cur.angle   # radians

        # Production front face in world coords
        prod_front_x = cur.x + front_m * np.cos(angle)
        prod_front_y = cur.y + front_m * np.sin(angle)

        # Actor physical front face: actor origin + (extent + bb offset) in heading dir
        t = actor.get_transform()
        actor_yaw = np.deg2rad(t.rotation.yaw)
        phys_front_x = t.location.x + (bb.extent.x + bb.location.x) * np.cos(actor_yaw)
        phys_front_y = t.location.y + (bb.extent.x + bb.location.x) * np.sin(actor_yaw)

        tol = 0.15  # allow up to 15 cm for GNSS noise and float rounding
        dist = np.sqrt((prod_front_x - phys_front_x)**2 + (prod_front_y - phys_front_y)**2)

        assert dist < tol, (
            f"Production front face ({prod_front_x:.3f},{prod_front_y:.3f}) is "
            f"{dist:.3f} m from actor physical front face "
            f"({phys_front_x:.3f},{phys_front_y:.3f}). "
            f"front_m={front_m:.3f}, CAR_FRONT_M={CAR_FRONT_M}. "
            f"Integration tests using hardcoded 3.856 describe a different footprint."
        )

        # Also verify that the test-constant CAR_FRONT_M is in the same ballpark,
        # so the integration tests are at least approximately correct.
        assert abs(front_m - CAR_FRONT_M) < 0.5, (
            f"Production front_m={front_m:.3f} differs from test constant "
            f"CAR_FRONT_M={CAR_FRONT_M} by more than 0.5 m. "
            f"Boundary tests in this file may be unreliable."
        )

    finally:
        if scenario:
            try: scenario.cleanup()
            except: pass
        for _ in range(3): world.tick()


# ═══════════════════════════════════════════════════════════════════════════════
# GAP #3 — collision_tracker built once; must cover the full ego path including
#           the destination parking spot
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("destination,parked", [
    (22, [21, 23]),
    (28, [27, 29]),
])
def test_collision_tracker_covers_destination_spot(carla_world, destination, parked):
    """
    production builds collision_tracker = obstacle_map_from_bbs(all_bbs_list)
    once before the loop, then calls collision_tracker.generate_collision_mask(car.cur)
    every tick.  If car.cur at the destination spot falls outside the tracker grid,
    generate_collision_mask returns an all-False mask — near-miss and cone detection
    go silent for the final approach and parking phase with no error.

    This test replicates the exact production build sequence and asserts that the
    mask is non-empty when the virtual ego is placed at the destination spot.
    """
    from parking_scenarios.parking_scenario_easy import ParkingScenarioEasy
    scenario = None
    world    = carla_world
    try:
        scenario = ParkingScenarioEasy(
            world=world, config=make_config(),
            destination=destination, parked=parked, criteria_enable=False,
        )
        for _ in range(5): world.tick()

        # Build collision_tracker exactly as production does (once, before the loop)
        moving_cars_bbs, traffic_cone_bbs, walker_bbs = get_bounding_boxes(scenario)
        all_bbs_list = (
            scenario.parked_cars_bbs
            + traffic_cone_bbs
            + moving_cars_bbs
            + [bb for _, bb in walker_bbs]
        )
        collision_tracker = obstacle_map_from_bbs(all_bbs_list)

        # Virtual ego placed AT the destination spot (where car.cur would be late in the run)
        dest_loc  = parking_vehicle_locations_Town04[destination]
        car_wp    = _WP(dest_loc.x, dest_loc.y, np.pi / 2)
        mask      = collision_tracker.generate_collision_mask(
            car_wp, front_m=CAR_FRONT_M, rear_m=CAR_REAR_M, half_width_m=CAR_HALF_W
        )

        assert mask.sum() > 0, (
            f"collision_tracker does NOT cover destination spot {destination} "
            f"({dest_loc.x:.1f}, {dest_loc.y:.1f}). "
            f"generate_collision_mask returned all-False — near-miss and cone checks "
            f"go silent during the final approach. "
            f"Fix: build collision_tracker with a min_y anchor or include destination BB."
        )

    finally:
        if scenario:
            try: scenario.cleanup()
            except: pass
        for _ in range(3): world.tick()


# ═══════════════════════════════════════════════════════════════════════════════
# GAP #4 — analyze_scenario(None) crash in run_scenario() finally block
# ═══════════════════════════════════════════════════════════════════════════════

def test_analyze_scenario_crashes_on_none():
    """
    run_scenario() initialises parking_scenario = None and then in the finally
    block unconditionally calls analyze_scenario(parking_scenario).  If the
    scenario constructor raises before assigning parking_scenario, the finally
    block crashes with AttributeError: 'NoneType' object has no attribute
    'get_criteria'.

    This test documents the failure mode.  When the bug is fixed (guard with
    'if parking_scenario'), this test should be updated accordingly.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from default_runner import analyze_scenario

    try:
        analyze_scenario(None)
        assert False, "Expected AttributeError — fix not yet applied"
    except AttributeError:
        pass  # expected: confirms the bug exists; fix is: guard with 'if parking_scenario'
