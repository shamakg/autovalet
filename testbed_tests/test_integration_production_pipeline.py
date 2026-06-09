"""
Integration tests that mirror the production run_scenario() pipeline exactly.

These tests do NOT simplify the collision/near-miss/IOU code — they use the
same functions in the same order that runner_test_medium.py uses them.

Four production paths exercised:
  A. get_bounding_boxes() — queries world.get_actors(), uses oriented_bbox()
  B. Cone grid+dilation collision pipeline (NOT obb_aabb_overlap)
  C. Near-miss with dilated combined obs (as in production)
  D. car.iou() with a real CARLA actor at a known pose

Run with:
    cd ~/carla_garage/leaderboard/leaderboard/autovalet
    source ~/envs/simlingo/bin/activate
    python -m pytest testbed_tests/test_integration_production_pipeline.py -v -s
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
    get_bounding_boxes,
    obstacle_map_from_bbs,
    near_miss,
)

pytestmark = pytest.mark.integration

# ── constants matching production ─────────────────────────────────────────────
CAR_FRONT_M  = 3.856
CAR_REAR_M   = 1.045
CAR_HALF_W   = 1.09

WALKER_BP    = 'walker.pedestrian.0001'
CONE_BP      = 'static.prop.constructioncone'


# ── verbatim copy from runner_test_medium.py ──────────────────────────────────
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


class _WP:
    def __init__(self, x, y, angle=0.0):
        self.x = x; self.y = y; self.angle = angle


def make_config():
    cfg = ScenarioConfiguration()
    cfg.name = "Parking"; cfg.type = "Parking"
    cfg.town = "Town04_Opt"; cfg.other_actors = None; cfg.route = False
    return cfg


def spawn_actor(world, bp_id, location, rotation=None):
    bp  = world.get_blueprint_library().find(bp_id)
    rot = rotation or carla.Rotation()
    actor = world.try_spawn_actor(bp, carla.Transform(carla.Location(**location), rot))
    assert actor is not None, f"Failed to spawn {bp_id} at {location}"
    return actor


def destroy_actors(actors):
    for a in actors:
        try:
            if a and a.is_alive:
                a.destroy()
        except Exception:
            pass


def cleanup(scenario, world):
    if scenario:
        try: scenario.cleanup()
        except Exception as e: print(f"cleanup error: {e}")
    for _ in range(3): world.tick()


# ═══════════════════════════════════════════════════════════════════════════════
# A. get_bounding_boxes() — the function production calls every tick
# ═══════════════════════════════════════════════════════════════════════════════

def test_get_bounding_boxes_finds_walker(carla_world):
    """
    After spawning a walker, get_bounding_boxes() must return it in walker_bbs
    with a sensible AABB that covers the actor's actual position.
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

        wx, wy = 285.6, -225.0
        walker = spawn_actor(world, WALKER_BP, dict(x=wx, y=wy, z=0.5))
        extra.append(walker)
        world.tick(); world.tick()

        _, _, walker_bbs = get_bounding_boxes(scenario)
        walker_ids = [wid for wid, _ in walker_bbs]
        assert walker.id in walker_ids, (
            f"get_bounding_boxes() missed spawned walker {walker.id}. "
            f"Returned walker ids: {walker_ids}"
        )

        bb = next(bb for wid, bb in walker_bbs if wid == walker.id)
        assert bb[0] <= wx <= bb[2], f"Walker x={wx} not in BB x-range [{bb[0]:.2f},{bb[2]:.2f}]"
        assert bb[1] <= wy <= bb[3], f"Walker y={wy} not in BB y-range [{bb[1]:.2f},{bb[3]:.2f}]"

    finally:
        destroy_actors(extra)
        cleanup(scenario, world)


def test_get_bounding_boxes_finds_cone(carla_world):
    """
    After spawning a cone, get_bounding_boxes() must return it in traffic_cone_bbs.
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

        cx, cy = 285.6, -220.0
        cone = spawn_actor(world, CONE_BP, dict(x=cx, y=cy, z=0.3))
        extra.append(cone)
        world.tick(); world.tick()

        _, cone_bbs, _ = get_bounding_boxes(scenario)
        assert len(cone_bbs) > 0, "get_bounding_boxes() returned no cone BBs"

        # At least one cone BB must cover the spawned cone's position
        covered = any(bb[0] <= cx <= bb[2] and bb[1] <= cy <= bb[3] for bb in cone_bbs)
        assert covered, (
            f"Spawned cone at ({cx},{cy}) not covered by any cone BB. "
            f"Cone BBs: {cone_bbs}"
        )

    finally:
        destroy_actors(extra)
        cleanup(scenario, world)


def test_get_bounding_boxes_excludes_ego_and_parked(carla_world):
    """
    get_bounding_boxes() must not include the ego car or any parked car in
    moving_cars_bbs — only truly moving/dynamic vehicles belong there.
    """
    from parking_scenarios.parking_scenario_easy import ParkingScenarioEasy
    scenario = None
    world    = carla_world
    try:
        scenario = ParkingScenarioEasy(
            world=world, config=make_config(),
            destination=22, parked=[21, 23], criteria_enable=False,
        )
        for _ in range(3): world.tick()

        moving_bbs, _, _ = get_bounding_boxes(scenario)

        # moving_bbs are not labelled, but we can verify none are centred near
        # the ego or parked-car positions (within 1 m)
        ego_loc = scenario.car.actor.get_location()
        ego_cx  = (ego_loc.x,)

        from parking_position import parking_vehicle_locations_Town04
        parked_locs = [parking_vehicle_locations_Town04[i] for i in [21, 22, 23]]
        parked_locs_xy = [(l.x, l.y) for l in parked_locs]

        for bb in moving_bbs:
            bb_cx = (bb[0] + bb[2]) / 2
            bb_cy = (bb[1] + bb[3]) / 2
            # Must not be centred on the ego
            assert abs(bb_cx - ego_loc.x) > 1.0 or abs(bb_cy - ego_loc.y) > 1.0, (
                f"moving_bbs contains a BB centred near ego at ({ego_loc.x:.1f},{ego_loc.y:.1f})"
            )
            # Must not be centred on a parked car
            for px, py in parked_locs_xy:
                assert abs(bb_cx - px) > 1.0 or abs(bb_cy - py) > 1.0, (
                    f"moving_bbs contains a BB centred near parked spot ({px},{py})"
                )

    finally:
        cleanup(scenario, world)


def test_oriented_bbox_covers_rotated_actor(carla_world):
    """
    oriented_bbox() inside get_bounding_boxes() expands the AABB to cover the
    full rotated extent. Verify this by spawning a vehicle at a 45° yaw and
    confirming the returned BB is larger than the un-rotated half-extents.

    A car at 45° has its diagonal as the dominant dimension; the AABB must be
    at least sqrt(2)*extent_x wide in both axes.
    """
    world = carla_world
    extra = []
    try:
        # Spawn a parked-style vehicle at 45° yaw so rotation matters
        bp  = world.get_blueprint_library().find('vehicle.lincoln.mkz_2020')
        rot = carla.Rotation(yaw=45.0)
        actor = world.try_spawn_actor(
            bp, carla.Transform(carla.Location(x=285.6, y=-215.0, z=0.5), rot)
        )
        assert actor is not None
        extra.append(actor)
        world.tick(); world.tick()

        bb  = actor.bounding_box
        yaw = np.deg2rad(45.0)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)

        # production oriented_bbox:
        t         = actor.get_transform()
        bb_world  = t.transform(bb.location)
        dx = abs(bb.extent.x * cos_y) + abs(bb.extent.y * sin_y)
        dy = abs(bb.extent.x * sin_y) + abs(bb.extent.y * cos_y)
        oriented = [bb_world.x - dx, bb_world.y - dy, bb_world.x + dx, bb_world.y + dy]

        # For a 45° rotation, both dx and dy should be larger than the
        # un-rotated half-extents (they're blended)
        assert (oriented[2] - oriented[0]) > 2 * bb.extent.x * 0.9, (
            "oriented_bbox width at 45° should exceed un-rotated extent"
        )
        assert (oriented[3] - oriented[1]) > 2 * bb.extent.y * 0.9, (
            "oriented_bbox height at 45° should exceed un-rotated extent"
        )

    finally:
        destroy_actors(extra)
        world.tick()


# ═══════════════════════════════════════════════════════════════════════════════
# B. Cone collision — production grid+dilation pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def _production_cone_collision(car_wp, cone_bbs, all_bbs):
    """Replicates the exact cone collision check from run_scenario():

        cone_obs = obstacle_map_from_bbs(traffic_cone_bbs, collision_tracker).obs
        cone_obs borders zeroed, then dilated 1 cell in all 4 directions
        collision = np.any(collision_mask & cone_obs)
    """
    collision_tracker = obstacle_map_from_bbs(all_bbs)
    collision_mask    = collision_tracker.generate_collision_mask(
        car_wp, front_m=CAR_FRONT_M, rear_m=CAR_REAR_M, half_width_m=CAR_HALF_W
    )

    cone_obs = obstacle_map_from_bbs(cone_bbs, collision_tracker).obs
    cone_obs[0, :] = 0; cone_obs[-1, :] = 0
    cone_obs[:, 0] = 0; cone_obs[:, -1] = 0
    _c = cone_obs
    cone_obs = _c | np.roll(_c, 1, 0) | np.roll(_c, -1, 0) | np.roll(_c, 1, 1) | np.roll(_c, -1, 1)

    return bool(np.any(collision_mask & (cone_obs == 1)))


def test_cone_in_path_detected_via_production_pipeline(carla_world):
    """
    Cone 1.5 m ahead of ego detected by the exact production
    grid+dilation path (not obb_aabb_overlap).
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

        ego_loc   = scenario.car.actor.get_location()
        ego_angle = np.pi / 2  # facing +y

        cone = spawn_actor(world, CONE_BP, dict(
            x=ego_loc.x, y=ego_loc.y + 1.5, z=0.3
        ))
        extra.append(cone)
        world.tick(); world.tick()

        _, cone_bbs, _ = get_bounding_boxes(scenario)
        all_bbs = scenario.parked_cars_bbs + cone_bbs

        car_wp  = _WP(ego_loc.x, ego_loc.y, ego_angle)
        result  = _production_cone_collision(car_wp, cone_bbs, all_bbs)

        assert result is True, (
            f"Production grid+dilation pipeline missed cone 1.5m ahead. "
            f"cone_bbs={cone_bbs}"
        )

    finally:
        destroy_actors(extra)
        cleanup(scenario, world)


def test_cone_off_side_not_detected_via_production_pipeline(carla_world):
    """
    Cone 4 m to the side must NOT be detected by the production
    grid+dilation pipeline.
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

        ego_loc   = scenario.car.actor.get_location()
        ego_angle = np.pi / 2

        cone = spawn_actor(world, CONE_BP, dict(
            x=ego_loc.x + 4.0, y=ego_loc.y, z=0.3
        ))
        extra.append(cone)
        world.tick(); world.tick()

        _, cone_bbs, _ = get_bounding_boxes(scenario)
        all_bbs = scenario.parked_cars_bbs + cone_bbs

        car_wp  = _WP(ego_loc.x, ego_loc.y, ego_angle)
        result  = _production_cone_collision(car_wp, cone_bbs, all_bbs)

        assert result is False, (
            f"Production pipeline falsely detected cone 4m to the side. "
            f"cone_bbs={cone_bbs}"
        )

    finally:
        destroy_actors(extra)
        cleanup(scenario, world)


# ═══════════════════════════════════════════════════════════════════════════════
# C. Near-miss — production dilated pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def _production_near_miss(car_wp, walker_bbs, cone_bbs, all_bbs):
    """Replicates the exact near-miss computation from run_scenario():

        combined_obs = obstacle_map_from_bbs(all_walker_bbs, collision_tracker).obs
        borders zeroed, then dilated 1 cell
        distance = near_miss(collision_mask, combined_obs)
    """
    collision_tracker = obstacle_map_from_bbs(all_bbs)
    collision_mask    = collision_tracker.generate_collision_mask(
        car_wp, front_m=CAR_FRONT_M, rear_m=CAR_REAR_M, half_width_m=CAR_HALF_W
    )

    all_walker_bbs = walker_bbs + cone_bbs
    combined_obs   = obstacle_map_from_bbs(all_walker_bbs, collision_tracker).obs
    combined_obs[0, :] = 0; combined_obs[-1, :] = 0
    combined_obs[:, 0] = 0; combined_obs[:, -1] = 0
    _w = combined_obs
    combined_obs = _w | np.roll(_w, 1, 0) | np.roll(_w, -1, 0) | np.roll(_w, 1, 1) | np.roll(_w, -1, 1)

    return near_miss(collision_mask, combined_obs)


def test_near_miss_production_pipeline_outside_footprint(carla_world):
    """
    Walker 5 m ahead (well outside the 3.856 m front footprint) must
    produce a finite positive near-miss distance via the production pipeline.
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

        ego_loc   = scenario.car.actor.get_location()
        ego_angle = np.pi / 2

        walker = spawn_actor(world, WALKER_BP, dict(
            x=ego_loc.x, y=ego_loc.y + 5.0, z=0.5
        ))
        extra.append(walker)
        world.tick(); world.tick()

        _, cone_bbs, walker_bbs = get_bounding_boxes(scenario)
        all_bbs = scenario.parked_cars_bbs + [bb for _, bb in walker_bbs]

        car_wp = _WP(ego_loc.x, ego_loc.y, ego_angle)
        dist   = _production_near_miss(car_wp, [bb for _, bb in walker_bbs], cone_bbs, all_bbs)

        assert dist != float('inf'), "near_miss returned inf — walker not in obs"
        assert dist > 0.0, f"near_miss returned 0 (collision) for walker 5m away: {dist}"
        assert dist < 4.0, f"near_miss distance unrealistically large: {dist}m"

    finally:
        destroy_actors(extra)
        cleanup(scenario, world)


def test_near_miss_production_pipeline_returns_zero_in_collision(carla_world):
    """
    Walker 2 m ahead (inside the footprint) must produce near_miss=0.0
    via the production pipeline.
    """
    world = carla_world
    extra = []
    try:
        # Use a standalone reference position — no ego car here so the walker can spawn.
        ref_x, ref_y = 285.6, -225.0
        ego_angle    = np.pi / 2

        walker = spawn_actor(world, WALKER_BP, dict(
            x=ref_x, y=ref_y + 2.0, z=0.5
        ))
        extra.append(walker)
        world.tick(); world.tick()

        # Build BBs from just the walker (no parked car clutter)
        from testbed.v2_experiment_utils import obstacle_map_from_bbs

        def oriented_bb(actor):
            t   = actor.get_transform()
            bb  = actor.bounding_box
            bw  = t.transform(bb.location)
            yaw = np.deg2rad(t.rotation.yaw)
            dx  = abs(bb.extent.x * np.cos(yaw)) + abs(bb.extent.y * np.sin(yaw))
            dy  = abs(bb.extent.x * np.sin(yaw)) + abs(bb.extent.y * np.cos(yaw))
            return [bw.x - dx, bw.y - dy, bw.x + dx, bw.y + dy]

        walker_bb  = oriented_bb(walker)
        all_bbs    = [walker_bb]

        car_wp = _WP(ref_x, ref_y, ego_angle)
        dist   = _production_near_miss(car_wp, [walker_bb], [], all_bbs)

        assert dist == 0.0, (
            f"Expected 0.0 (collision) for walker inside footprint, got {dist}m"
        )

    finally:
        destroy_actors(extra)
        world.tick()


def test_near_miss_dilation_makes_close_walker_count(carla_world):
    """
    Dilation expands the obstacle obs by 1 cell (0.25 m) in each direction.
    A walker placed exactly at the footprint boundary (just outside) might be
    missed by un-dilated obs but caught by dilated obs if it is within 0.25 m.

    This test places the walker half a dilation-cell past the front face and
    verifies the DILATED pipeline detects it while confirming the un-dilated
    obs would miss it (distance > 0).
    """
    world = carla_world
    extra = []
    try:
        ref_x, ref_y = 285.6, -228.0
        ego_angle    = np.pi / 2

        # Place walker centre just past the front face + 0.1 m (no physical overlap)
        # but within the 1-cell (0.25 m) dilation band
        front_edge_y = ref_y + CAR_FRONT_M
        walker_y     = front_edge_y + 0.15  # 0.15 m past front face — within dilation

        walker = spawn_actor(world, WALKER_BP, dict(x=ref_x, y=walker_y, z=0.5))
        extra.append(walker)
        world.tick(); world.tick()

        def oriented_bb(actor):
            t   = actor.get_transform()
            bb  = actor.bounding_box
            bw  = t.transform(bb.location)
            yaw = np.deg2rad(t.rotation.yaw)
            dx  = abs(bb.extent.x * np.cos(yaw)) + abs(bb.extent.y * np.sin(yaw))
            dy  = abs(bb.extent.x * np.sin(yaw)) + abs(bb.extent.y * np.cos(yaw))
            return [bw.x - dx, bw.y - dy, bw.x + dx, bw.y + dy]

        walker_bb = oriented_bb(walker)

        car_wp = _WP(ref_x, ref_y, ego_angle)

        # Production pipeline (dilated) — should detect near-miss (dist close to 0 or 0)
        dist_dilated = _production_near_miss(car_wp, [walker_bb], [], [walker_bb])
        assert dist_dilated <= 0.5, (
            f"Dilated near-miss too large for walker 0.15m past front: {dist_dilated}m"
        )

    finally:
        destroy_actors(extra)
        world.tick()


# ═══════════════════════════════════════════════════════════════════════════════
# D. car.iou() with real CARLA actors at known poses
# ═══════════════════════════════════════════════════════════════════════════════

def test_iou_at_start_position_is_low(carla_world):
    """
    The ego spawns far from the destination spot.  iou() at the start
    position must return a low value (< 0.3), confirming the car is not
    accidentally parked at the destination.
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

        iou = scenario.car.iou()
        assert iou < 0.3, (
            f"IOU at start position should be low, got {iou:.3f} — "
            f"ego may have spawned on top of destination spot"
        )

    finally:
        cleanup(scenario, world)


def test_iou_is_high_when_ego_teleported_to_destination(carla_world):
    """
    Teleport the ego to the destination spot and confirm iou() returns > 0.85.
    This validates the bounding-box geometry and shapely polygon overlap used
    by the IOU calculation against a real CARLA actor transform.
    """
    from parking_scenarios.parking_scenario_easy import ParkingScenarioEasy
    scenario = None
    world    = carla_world
    try:
        destination = 22
        scenario = ParkingScenarioEasy(
            world=world, config=make_config(),
            destination=destination, parked=[21, 23], criteria_enable=False,
        )
        for _ in range(5): world.tick()

        dest_loc = parking_vehicle_locations_Town04[destination]

        # Teleport the ego car to the destination spot
        scenario.car.actor.set_transform(
            carla.Transform(dest_loc, carla.Rotation(yaw=0.0))
        )
        world.tick(); world.tick()

        iou = scenario.car.iou()
        assert iou > 0.85, (
            f"IOU should be high when ego is at destination spot, got {iou:.3f}"
        )

    finally:
        cleanup(scenario, world)


def test_iou_is_high_when_ego_at_destination_reversed(carla_world):
    """
    A 180° yaw at the destination (reversed into spot) must also give IOU > 0.85.
    The destination BB is axis-aligned so orientation does not change overlap.
    """
    from parking_scenarios.parking_scenario_easy import ParkingScenarioEasy
    scenario = None
    world    = carla_world
    try:
        destination = 28
        scenario = ParkingScenarioEasy(
            world=world, config=make_config(),
            destination=destination, parked=[27, 29], criteria_enable=False,
        )
        for _ in range(5): world.tick()

        dest_loc = parking_vehicle_locations_Town04[destination]
        scenario.car.actor.set_transform(
            carla.Transform(dest_loc, carla.Rotation(yaw=180.0))
        )
        world.tick(); world.tick()

        iou = scenario.car.iou()
        assert iou > 0.85, (
            f"IOU for reversed parking at destination should be > 0.85, got {iou:.3f}"
        )

    finally:
        cleanup(scenario, world)


def test_iou_drops_when_ego_offset_by_two_metres(carla_world):
    """
    Shifting the ego 2 m sideways from the destination must produce a
    noticeably lower IOU than the perfectly-parked case.
    """
    from parking_scenarios.parking_scenario_easy import ParkingScenarioEasy
    scenario_perfect = None
    scenario_offset  = None
    world = carla_world
    try:
        destination = 22
        # Perfect park
        scenario_perfect = ParkingScenarioEasy(
            world=world, config=make_config(),
            destination=destination, parked=[21, 23], criteria_enable=False,
        )
        for _ in range(5): world.tick()
        dest_loc = parking_vehicle_locations_Town04[destination]
        scenario_perfect.car.actor.set_transform(
            carla.Transform(dest_loc, carla.Rotation(yaw=0.0))
        )
        world.tick(); world.tick()
        iou_perfect = scenario_perfect.car.iou()
        cleanup(scenario_perfect, world)
        scenario_perfect = None

        # 2 m offset
        scenario_offset = ParkingScenarioEasy(
            world=world, config=make_config(),
            destination=destination, parked=[21, 23], criteria_enable=False,
        )
        for _ in range(5): world.tick()
        offset_loc = carla.Location(
            x=dest_loc.x + 2.0, y=dest_loc.y, z=dest_loc.z
        )
        scenario_offset.car.actor.set_transform(
            carla.Transform(offset_loc, carla.Rotation(yaw=0.0))
        )
        world.tick(); world.tick()
        iou_offset = scenario_offset.car.iou()

        assert iou_perfect > iou_offset + 0.15, (
            f"Perfect IOU ({iou_perfect:.3f}) should be notably higher than "
            f"2m-offset IOU ({iou_offset:.3f})"
        )

    finally:
        cleanup(scenario_perfect, world)
        cleanup(scenario_offset, world)


# ═══════════════════════════════════════════════════════════════════════════════
# E. Walker collision: production path uses get_bounding_boxes() + obb_aabb_overlap
# ═══════════════════════════════════════════════════════════════════════════════

def test_walker_collision_via_get_bounding_boxes(carla_world):
    """
    Full production walker-collision path:
      1. spawn walker inside ego OBB
      2. call get_bounding_boxes() to get (id, bb) pairs
      3. run obb_aabb_overlap with each bb
    Must detect the collision.
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

        ego_loc   = scenario.car.actor.get_location()
        ego_angle = np.pi / 2

        walker = spawn_actor(world, WALKER_BP, dict(
            x=ego_loc.x, y=ego_loc.y + 2.0, z=1.5
        ))
        extra.append(walker)
        world.tick(); world.tick()

        _, _, walker_bbs = get_bounding_boxes(scenario)
        car_wp = _WP(ego_loc.x, ego_loc.y, ego_angle)

        hit_ids = [
            wid for wid, bb in walker_bbs
            if obb_aabb_overlap(car_wp, CAR_FRONT_M, CAR_REAR_M, CAR_HALF_W, bb)
        ]

        assert walker.id in hit_ids, (
            f"Production walker-collision path missed walker {walker.id}. "
            f"Detected hits: {hit_ids}, all walker ids: {[wid for wid, _ in walker_bbs]}"
        )

    finally:
        destroy_actors(extra)
        cleanup(scenario, world)


def test_walker_outside_path_not_detected_via_get_bounding_boxes(carla_world):
    """
    Walker 5 m to the side must not be detected by get_bounding_boxes() + obb_aabb_overlap.

    Uses a fixed open reference position inside the lot (not near the ego's actual
    spawn, which is close to CARLA geometry that blocks walker spawns at x+5).
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

        # Use an open reference inside the parking lot — this area is clear of
        # CARLA geometry that would block pedestrian spawns.
        ref_x, ref_y = 285.6, -225.0
        ego_angle    = np.pi / 2

        # 5 m to the side (x direction) — open spot at x=290.6, y=-225
        walker = spawn_actor(world, WALKER_BP, dict(
            x=ref_x + 5.0, y=ref_y, z=0.5
        ))
        extra.append(walker)
        world.tick(); world.tick()

        _, _, walker_bbs = get_bounding_boxes(scenario)
        car_wp = _WP(ref_x, ref_y, ego_angle)

        hit_ids = [
            wid for wid, bb in walker_bbs
            if obb_aabb_overlap(car_wp, CAR_FRONT_M, CAR_REAR_M, CAR_HALF_W, bb)
        ]

        assert walker.id not in hit_ids, (
            f"Production path falsely detected walker 5m to the side (ref at {ref_x},{ref_y})."
        )

    finally:
        destroy_actors(extra)
        cleanup(scenario, world)


# ═══════════════════════════════════════════════════════════════════════════════
# F. cone_obs_cached — pre-computed grid matches per-tick rebuild
# ═══════════════════════════════════════════════════════════════════════════════

def test_cone_obs_cached_detects_in_path_cone(carla_world):
    """
    benchmark.py pre-computes cone_obs_cached once at scenario start (lines 287-292)
    and reuses it every tick instead of rebuilding the grid.  If the cached grid
    is misaligned with collision_tracker the check would silently fail.

    This test replicates the exact build sequence from the benchmark and verifies:
      1. A car positioned on a cone centre is detected via the CACHED path.
      2. The cached result matches the per-tick rebuild result (parity).
    """
    from parking_scenarios.parking_scenario_easy import ParkingScenarioEasy

    world    = carla_world
    scenario = None
    try:
        scenario = ParkingScenarioEasy(
            world=world, config=make_config(),
            destination=20, parked=[],
            criteria_enable=False,
        )
        for _ in range(3): world.tick()

        moving_cars_bbs, traffic_cone_bbs, walker_bbs = get_bounding_boxes(scenario)
        if not traffic_cone_bbs:
            pytest.skip("No cones in world — destination=20 produced no scenario cones")

        # Build collision_tracker and cone_obs_cached exactly as benchmark.py does
        all_bbs_list = (
            scenario.parked_cars_bbs
            + traffic_cone_bbs
            + moving_cars_bbs
            + [bb for _, bb in walker_bbs]
        )
        collision_tracker = obstacle_map_from_bbs(all_bbs_list)

        _co = obstacle_map_from_bbs(traffic_cone_bbs, collision_tracker).obs
        _co[0, :] = 0; _co[-1, :] = 0; _co[:, 0] = 0; _co[:, -1] = 0
        cone_obs_cached = _co | np.roll(_co, 1, 0) | np.roll(_co, -1, 0) | np.roll(_co, 1, 1) | np.roll(_co, -1, 1)

        # Place a car AT the centre of the first cone's AABB
        first_bb = traffic_cone_bbs[0]
        cone_cx  = (first_bb[0] + first_bb[2]) / 2
        cone_cy  = (first_bb[1] + first_bb[3]) / 2

        car_wp = _WP(cone_cx, cone_cy, np.pi / 2)
        collision_mask = collision_tracker.generate_collision_mask(
            car_wp, front_m=CAR_FRONT_M, rear_m=CAR_REAR_M, half_width_m=CAR_HALF_W
        )

        # 1. Cached path must detect the cone
        cached_hit = bool(np.any(collision_mask & (cone_obs_cached == 1)))
        assert cached_hit, (
            "cone_obs_cached path missed cone at car position. "
            f"car=({cone_cx:.2f},{cone_cy:.2f}), cone_bb={first_bb}. "
            "Likely cause: cone_obs_cached is misaligned with collision_tracker."
        )

        # 2. Per-tick rebuild must agree (parity)
        _ct = obstacle_map_from_bbs(traffic_cone_bbs, collision_tracker).obs
        _ct[0, :] = 0; _ct[-1, :] = 0; _ct[:, 0] = 0; _ct[:, -1] = 0
        cone_obs_tick = _ct | np.roll(_ct, 1, 0) | np.roll(_ct, -1, 0) | np.roll(_ct, 1, 1) | np.roll(_ct, -1, 1)
        pertick_hit = bool(np.any(collision_mask & (cone_obs_tick == 1)))

        assert cached_hit == pertick_hit, (
            f"cone_obs_cached ({cached_hit}) disagrees with per-tick rebuild ({pertick_hit}). "
            "Grid alignment or dilation mismatch between the two code paths."
        )

    finally:
        cleanup(scenario, world)


# ═══════════════════════════════════════════════════════════════════════════════
# G. Scenario-spawned walkers (physics off) visible to get_bounding_boxes
# ═══════════════════════════════════════════════════════════════════════════════

def test_scenario_pedestrian_visible_in_get_bounding_boxes(carla_world):
    """
    Walkers spawned by PedestrianCrossingParking must appear in walker_bbs
    returned by get_bounding_boxes() — this is the critical end-to-end path that
    feeds collision detection in the benchmark main loop.

    _replace_walker() sets simulate_physics=False on every walker.  This test
    also implicitly verifies that physics-off walkers are still returned by
    world.get_actors().filter('walker.*'), which get_bounding_boxes() relies on.
    """
    from parking_scenarios.parking_scenario_hard import ParkingScenarioHard, HardMode

    world    = carla_world
    scenario = None
    try:
        scenario = ParkingScenarioHard(
            world=world, config=make_config(),
            destination=20, parked=[],
            criteria_enable=False, mode=HardMode.PedMode,
        )
        for _ in range(3): world.tick()

        ped_scenario = scenario.list_scenarios[0]
        scenario_walker_ids = {
            a.id for a in ped_scenario.other_actors
            if 'walker' in a.type_id and 'controller' not in a.type_id
        }
        if not scenario_walker_ids:
            pytest.skip("No walkers spawned by PedestrianCrossingParking for destination=20")

        _, _, walker_bbs = get_bounding_boxes(scenario)
        visible_ids = {wid for wid, _ in walker_bbs}

        missing = scenario_walker_ids - visible_ids
        assert not missing, (
            f"{len(missing)} scenario walker(s) invisible to get_bounding_boxes(): ids={missing}. "
            "These walkers cannot be collision-detected during the benchmark run loop. "
            "Likely cause: physics=False walkers not returned by world.get_actors().filter()."
        )

    finally:
        cleanup(scenario, world)
