"""
Dynamic collision integration tests — actors that intentionally enter and exit
the ego footprint, plus the CARLA vehicle collision sensor.

These tests exercise the full production state machine from runner_test_medium.py
with REAL moving actors (teleported through the ego path each CARLA tick), not
just static placement.  They verify:

  1. Walker teleported INTO ego footprint → obb_aabb_overlap fires
  2. Walker teleported OUT of footprint → obb_aabb_overlap clears
  3. Grace-tick: re-entry within GRACE_TICKS → counted ONCE (no double-count)
  4. Grace-tick: re-entry after full GRACE_TICKS expiry → counted TWICE
  5. Two simultaneous walkers: only the one in-path is counted
  6. Cone dilation pipeline fires when cone is in path
  7. CARLA vehicle collision sensor (vehicle_criterion.actual_value) fires on
     actual physical impact — this is the sensor that drives collisions_ref[0]
     in run_scenario()

Run with:
    cd ~/carla_garage/leaderboard/leaderboard/autovalet
    source ~/envs/simlingo/bin/activate
    python -m pytest testbed_tests/test_integration_dynamic_collisions.py -v -s
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import carla
import numpy as np
import pytest
import time

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration
from srunner.scenariomanager.timer import GameTime

from testbed.v2_experiment_utils import get_bounding_boxes, obstacle_map_from_bbs, near_miss

pytestmark = pytest.mark.integration

# ── constants ─────────────────────────────────────────────────────────────────
CAR_FRONT_M = 3.856
CAR_REAR_M  = 1.045
CAR_HALF_W  = 1.09
GRACE_TICKS = 20

WALKER_BP = 'walker.pedestrian.0001'
CONE_BP   = 'static.prop.constructioncone'


# ── verbatim from runner_test_medium.py ───────────────────────────────────────
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
            if a and a.is_alive: a.destroy()
        except Exception: pass


def cleanup(scenario, world):
    if scenario:
        try: scenario.cleanup()
        except Exception as e: print(f"cleanup: {e}")
    for _ in range(3): world.tick()


# ── production state-machine helpers ─────────────────────────────────────────

def one_tick_of_state_machine(walker_ids_this_tick, colliding_walker_ids, walker_last_seen):
    """Run exactly one tick of the production collision state machine.

    Returns the number of NEW collisions this tick (what production adds to
    walker_collisions_ref[0]).  Mutates colliding_walker_ids and walker_last_seen
    in place, exactly as run_scenario() does.
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
    colliding_walker_ids |= walker_ids_this_tick
    return len(new_collisions)


# ── production cone-collision helper ─────────────────────────────────────────

def production_cone_hit(car_wp, cone_bbs, collision_tracker):
    """Exact production cone grid+dilation check."""
    collision_mask = collision_tracker.generate_collision_mask(
        car_wp, front_m=CAR_FRONT_M, rear_m=CAR_REAR_M, half_width_m=CAR_HALF_W
    )
    cone_obs = obstacle_map_from_bbs(cone_bbs, collision_tracker).obs
    cone_obs[0,:]=0; cone_obs[-1,:]=0; cone_obs[:,0]=0; cone_obs[:,-1]=0
    _c = cone_obs
    cone_obs = _c | np.roll(_c,1,0)|np.roll(_c,-1,0)|np.roll(_c,1,1)|np.roll(_c,-1,1)
    return bool(np.any(collision_mask & (cone_obs == 1)))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Walker enters footprint → detection fires; exits → clears
# ═══════════════════════════════════════════════════════════════════════════════

def test_walker_enters_footprint_detected_then_clears(carla_world):
    """
    Teleport a walker INSIDE the ego footprint → obb_aabb_overlap must fire.
    Teleport the SAME walker OUTSIDE       → obb_aabb_overlap must clear.

    This verifies get_bounding_boxes() correctly tracks position changes between
    ticks — critical because production calls it every loop iteration.
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
        ego_angle    = np.pi / 2
        car_wp       = _WP(ref_x, ref_y, ego_angle)

        # Spawn outside first so there's a valid actor to move
        walker = spawn_actor(world, WALKER_BP, dict(x=ref_x + 6.0, y=ref_y, z=0.5))
        extra.append(walker)
        world.tick(); world.tick()

        # ── Move INSIDE (2 m ahead, well within CAR_FRONT_M=3.856) ──
        walker.set_transform(carla.Transform(
            carla.Location(x=ref_x, y=ref_y + 2.0, z=0.5)
        ))
        world.tick(); world.tick()

        _, _, walker_bbs = get_bounding_boxes(scenario)
        ids_inside = {wid for wid, bb in walker_bbs
                      if obb_aabb_overlap(car_wp, CAR_FRONT_M, CAR_REAR_M, CAR_HALF_W, bb)}
        assert walker.id in ids_inside, (
            f"Walker teleported inside footprint not detected. "
            f"Walker BB after move: "
            f"{next((bb for wid,bb in walker_bbs if wid==walker.id), None)}"
        )

        # ── Move OUTSIDE (6 m to the side) ──
        walker.set_transform(carla.Transform(
            carla.Location(x=ref_x + 6.0, y=ref_y, z=0.5)
        ))
        world.tick(); world.tick()

        _, _, walker_bbs2 = get_bounding_boxes(scenario)
        ids_outside = {wid for wid, bb in walker_bbs2
                       if obb_aabb_overlap(car_wp, CAR_FRONT_M, CAR_REAR_M, CAR_HALF_W, bb)}
        assert walker.id not in ids_outside, (
            f"Walker teleported outside footprint still detected."
        )

    finally:
        destroy_actors(extra)
        cleanup(scenario, world)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Single crossing counted exactly once
# ═══════════════════════════════════════════════════════════════════════════════

def test_single_crossing_counted_once(carla_world):
    """
    Walker enters footprint for 5 ticks → exits → state machine must count
    exactly 1 collision (not one per tick inside).
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
        ego_angle    = np.pi / 2
        car_wp       = _WP(ref_x, ref_y, ego_angle)

        walker = spawn_actor(world, WALKER_BP, dict(x=ref_x + 6.0, y=ref_y, z=0.5))
        extra.append(walker)
        world.tick(); world.tick()

        colliding_walker_ids = set()
        walker_last_seen     = {}
        total_collisions     = 0

        # 5 ticks INSIDE
        walker.set_transform(carla.Transform(carla.Location(x=ref_x, y=ref_y + 2.0, z=0.5)))
        for _ in range(5):
            world.tick()
            _, _, wbs = get_bounding_boxes(scenario)
            ids = {wid for wid, bb in wbs
                   if obb_aabb_overlap(car_wp, CAR_FRONT_M, CAR_REAR_M, CAR_HALF_W, bb)}
            total_collisions += one_tick_of_state_machine(ids, colliding_walker_ids, walker_last_seen)

        # 3 ticks OUTSIDE
        walker.set_transform(carla.Transform(carla.Location(x=ref_x + 6.0, y=ref_y, z=0.5)))
        for _ in range(3):
            world.tick()
            _, _, wbs = get_bounding_boxes(scenario)
            ids = {wid for wid, bb in wbs
                   if obb_aabb_overlap(car_wp, CAR_FRONT_M, CAR_REAR_M, CAR_HALF_W, bb)}
            total_collisions += one_tick_of_state_machine(ids, colliding_walker_ids, walker_last_seen)

        assert total_collisions == 1, (
            f"Expected exactly 1 collision for one crossing, got {total_collisions}"
        )

    finally:
        destroy_actors(extra)
        cleanup(scenario, world)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Grace-tick: re-entry within window → NOT recounted
# ═══════════════════════════════════════════════════════════════════════════════

def test_reentry_within_grace_not_recounted(carla_world):
    """
    Walker enters (3 ticks) → exits (10 ticks, < GRACE_TICKS=20) → re-enters.
    Must count only 1 collision total — the re-entry is within the grace window.
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

        walker = spawn_actor(world, WALKER_BP, dict(x=ref_x + 6.0, y=ref_y, z=0.5))
        extra.append(walker)
        world.tick(); world.tick()

        colliding_walker_ids = set()
        walker_last_seen     = {}
        total_collisions     = 0

        def run_ticks(n, inside):
            nonlocal total_collisions
            pos = carla.Location(x=ref_x, y=ref_y+2.0, z=0.5) if inside \
                  else carla.Location(x=ref_x+6.0, y=ref_y, z=0.5)
            walker.set_transform(carla.Transform(pos))
            for _ in range(n):
                world.tick()
                _, _, wbs = get_bounding_boxes(scenario)
                ids = {wid for wid, bb in wbs
                       if obb_aabb_overlap(car_wp, CAR_FRONT_M, CAR_REAR_M, CAR_HALF_W, bb)}
                total_collisions += one_tick_of_state_machine(ids, colliding_walker_ids, walker_last_seen)

        run_ticks(3, inside=True)     # enters
        run_ticks(10, inside=False)   # exits (10 < GRACE_TICKS)
        run_ticks(3, inside=True)     # re-enters within grace

        assert total_collisions == 1, (
            f"Re-entry within GRACE_TICKS={GRACE_TICKS} should not be recounted. "
            f"Got {total_collisions} collisions."
        )

    finally:
        destroy_actors(extra)
        cleanup(scenario, world)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Grace-tick: re-entry AFTER full expiry → IS recounted
# ═══════════════════════════════════════════════════════════════════════════════

def test_reentry_after_grace_expiry_recounted(carla_world):
    """
    Walker enters (3 ticks) → exits (GRACE_TICKS+2 ticks) → re-enters.
    Must count 2 collisions total — grace window has fully expired.
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

        walker = spawn_actor(world, WALKER_BP, dict(x=ref_x + 6.0, y=ref_y, z=0.5))
        extra.append(walker)
        world.tick(); world.tick()

        colliding_walker_ids = set()
        walker_last_seen     = {}
        total_collisions     = 0

        def run_ticks(n, inside):
            nonlocal total_collisions
            pos = carla.Location(x=ref_x, y=ref_y+2.0, z=0.5) if inside \
                  else carla.Location(x=ref_x+6.0, y=ref_y, z=0.5)
            walker.set_transform(carla.Transform(pos))
            for _ in range(n):
                world.tick()
                _, _, wbs = get_bounding_boxes(scenario)
                ids = {wid for wid, bb in wbs
                       if obb_aabb_overlap(car_wp, CAR_FRONT_M, CAR_REAR_M, CAR_HALF_W, bb)}
                total_collisions += one_tick_of_state_machine(ids, colliding_walker_ids, walker_last_seen)

        run_ticks(3, inside=True)              # first crossing
        run_ticks(GRACE_TICKS + 2, inside=False)  # wait until grace expires
        run_ticks(3, inside=True)              # second crossing — must recount

        assert total_collisions == 2, (
            f"Expected 2 collisions after grace expiry, got {total_collisions}"
        )

    finally:
        destroy_actors(extra)
        cleanup(scenario, world)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Two walkers simultaneously — only the in-path one is counted
# ═══════════════════════════════════════════════════════════════════════════════

def test_two_walkers_only_crossing_one_counted(carla_world):
    """
    Walker A: teleported into ego footprint.
    Walker B: stays 5 m to the side throughout.

    The state machine must count 1 collision (walker A), not 2.
    Walker B must never appear in the colliding set.
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

        walker_a = spawn_actor(world, WALKER_BP, dict(x=ref_x,     y=ref_y+2.0, z=0.5))
        walker_b = spawn_actor(world, WALKER_BP, dict(x=ref_x+5.0, y=ref_y,     z=0.5))
        extra    = [walker_a, walker_b]
        world.tick(); world.tick()

        colliding_walker_ids = set()
        walker_last_seen     = {}
        total_collisions     = 0
        b_ever_colliding     = False

        for _ in range(5):
            world.tick()
            _, _, wbs = get_bounding_boxes(scenario)
            ids = {wid for wid, bb in wbs
                   if obb_aabb_overlap(car_wp, CAR_FRONT_M, CAR_REAR_M, CAR_HALF_W, bb)}
            if walker_b.id in ids:
                b_ever_colliding = True
            total_collisions += one_tick_of_state_machine(ids, colliding_walker_ids, walker_last_seen)

        assert not b_ever_colliding, "Walker B (to the side) was incorrectly flagged as colliding"
        assert total_collisions == 1, (
            f"Expected 1 collision (walker A only), got {total_collisions}"
        )
        assert walker_a.id in colliding_walker_ids, "Walker A not in colliding set"

    finally:
        destroy_actors(extra)
        cleanup(scenario, world)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Cone: enters and exits ego path — production pipeline fires/clears
# ═══════════════════════════════════════════════════════════════════════════════

def test_cone_enters_and_exits_path_via_production_pipeline(carla_world):
    """
    Cone teleported INTO ego path → production grid+dilation fires.
    Cone teleported OUT of ego path → pipeline clears.
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

        cone = spawn_actor(world, CONE_BP, dict(x=ref_x + 5.0, y=ref_y, z=0.3))
        extra.append(cone)
        world.tick(); world.tick()

        # ── Cone OUTSIDE → should NOT fire ──
        _, cone_bbs, _ = get_bounding_boxes(scenario)
        all_bbs = scenario.parked_cars_bbs + cone_bbs
        ct = obstacle_map_from_bbs(all_bbs)
        assert not production_cone_hit(car_wp, cone_bbs, ct), \
            "Cone outside path falsely detected"

        # ── Move cone INTO path (1.5 m ahead) ──
        cone.set_transform(carla.Transform(carla.Location(x=ref_x, y=ref_y + 1.5, z=0.3)))
        world.tick(); world.tick()

        _, cone_bbs2, _ = get_bounding_boxes(scenario)
        all_bbs2 = scenario.parked_cars_bbs + cone_bbs2
        ct2 = obstacle_map_from_bbs(all_bbs2)
        assert production_cone_hit(car_wp, cone_bbs2, ct2), \
            f"Cone moved into path NOT detected. cone_bbs={cone_bbs2}"

        # ── Move cone back out ──
        cone.set_transform(carla.Transform(carla.Location(x=ref_x + 5.0, y=ref_y, z=0.3)))
        world.tick(); world.tick()

        _, cone_bbs3, _ = get_bounding_boxes(scenario)
        all_bbs3 = scenario.parked_cars_bbs + cone_bbs3
        ct3 = obstacle_map_from_bbs(all_bbs3)
        assert not production_cone_hit(car_wp, cone_bbs3, ct3), \
            "Cone moved back out still detected"

    finally:
        destroy_actors(extra)
        cleanup(scenario, world)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CARLA vehicle collision sensor — fires on actual physical impact
# ═══════════════════════════════════════════════════════════════════════════════

def test_vehicle_collision_sensor_fires_on_physical_impact(carla_world):
    """
    Drives a vehicle into the stationary ego using CARLA physics.
    CollisionTest.actual_value (the exact field production reads via
    vehicle_criterion.actual_value) must increment to > 0.

    The impactor is given a velocity toward the ego so CARLA's collision sensor
    fires. The ego needs any non-zero velocity for the sensor to count — we give
    it a tiny shove at the moment of impact.

    Note: CARLA's CollisionTest ignores collisions when the ego is stationary
    (speed < 0.1 m/s), so we briefly move the ego toward the impactor.
    """
    from parking_scenarios.parking_scenario_easy import ParkingScenarioEasy
    from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
    scenario = None
    world    = carla_world
    extra    = []
    try:
        scenario = ParkingScenarioEasy(
            world=world, config=make_config(),
            destination=22, parked=[21, 23], criteria_enable=False,
        )
        for _ in range(3): world.tick()

        ego = scenario.car.actor

        # Build a VehicleCollisionTest criterion exactly as production does
        vehicle_criterion = CollisionTest(
            ego, other_actor_type="vehicle", name="VehicleCollisionTest"
        )
        vehicle_criterion.initialise()
        world.tick(); world.tick()

        # Spawn the impactor 6 m ahead of the ego (+y direction)
        ego_loc = ego.get_location()
        bp = world.get_blueprint_library().find('vehicle.tesla.model3')
        impactor = world.try_spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=ego_loc.x, y=ego_loc.y + 6.0, z=0.3),
                carla.Rotation(yaw=270.0)   # facing -y (toward ego)
            )
        )
        assert impactor is not None, "Could not spawn impactor vehicle"
        extra.append(impactor)
        world.tick(); world.tick()

        # Give both vehicles physics and velocity so the sensor registers the hit
        ego.set_simulate_physics(True)
        impactor.set_simulate_physics(True)
        world.tick()

        # Ego moves slightly toward impactor; impactor drives into ego
        ego.set_target_velocity(carla.Vector3D(0,  2.0, 0))   # +y (toward impactor)
        impactor.set_target_velocity(carla.Vector3D(0, -8.0, 0))  # -y (toward ego)

        # Tick until collision sensor fires or 60 ticks pass (~3 s at 20 Hz)
        for _ in range(60):
            world.tick()
            timestamp = world.get_snapshot().timestamp
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()
            vehicle_criterion.update()
            if vehicle_criterion.actual_value > 0:
                break

        assert vehicle_criterion.actual_value > 0, (
            "Vehicle CollisionTest criterion did not fire after impactor drove into ego. "
            "This is the exact sensor production reads via vehicle_criterion.actual_value."
        )

    finally:
        vehicle_criterion.terminate(None)
        destroy_actors(extra)
        cleanup(scenario, world)
