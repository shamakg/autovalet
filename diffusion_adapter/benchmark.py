### Version 3 of the code, with the goal being to renew the pedestrian crossing abstraction

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime
from enum import Enum

sys.path.insert(0, '/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet')

import carla
import matplotlib.pyplot as plt
import numpy as np
import py_trees

from srunner.scenariomanager.timer import GameTime
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from parking_scenarios.parking_scenario_easy import ParkingScenarioEasy
from parking_scenarios.parking_scenario_medium import ParkingScenarioMedium
from parking_scenarios.parking_scenario_hard import HardMode, ParkingScenarioHard
from parking_scenarios.opposite_vehicle_parking import CollisionMode
from v2 import TrajectoryPoint
from v2_experiment import SCENARIOS
from testbed.v2_experiment_utils import (
    get_bounding_boxes,
    load_client,
    near_miss,
    obstacle_map_from_bbs,
    town04_load,
    town04_spectator_bev,
)
from testbed.recording_utils import (
    ego_to_world,
    finalize_recording,
    init_recording,
    make_cleanup_handler,
    process_recording_frames,
    save_trajectory_plot,
)
from parking_position import parking_vehicle_locations_Town04
from agent_interface import DiffusionAdapter

# ---------------------------------------------------------------
# ---------------------------------------------------------------
### Many of these are legacy
NUM_RANDOM_CARS = 50
NETWORK_SEND_LATENCIES = [0]        # ms
PERCEPTION_LATENCY = 200            # ms
SMALL_PERCEPTION_LATENCY = 100      # ms
RECV_LATENCY = 100                  # ms
MODE_PERIOD = 500                   # ms
PLANNING_PERIOD = 500               # ms
DATA_COLLECTION_PERIOD = 200        # ms
SHOULD_PIPELINE = True
SHOULD_ADJUST_MODEL = False
TIMEOUT = 120 * 1000                # ms
IMAGE_DOWNSIZE = 1
WALL_TIMEOUT = 290                  # s
RECORD = True

NETWORK_SEND_LATENCIES = [latency // IMAGE_DOWNSIZE for latency in NETWORK_SEND_LATENCIES]
# ---------------------------------------------------------------
# ---------------------------------------------------------------

def distance_point_to_segment(p, s1, s2):
    v = s2 - s1
    w = p - s1
    c1 = np.dot(w, v)
    if c1 <= 0:
        return np.linalg.norm(p - s1)
    c2 = np.dot(v, v)
    if c2 <= c1:
        return np.linalg.norm(p - s2)
    b = c1 / c2
    pb = s1 + b * v
    return np.linalg.norm(p - pb)

def calculate_min_distance_to_door(ego_actor, door_vehicle):
    # Ego bounding box corners in world coordinates
    ego_trans = ego_actor.get_transform()
    ego_ext = ego_actor.bounding_box.extent
    # Corners in local coords: (-x, -y), (-x, +y), (+x, -y), (+x, +y)
    ego_corners_local = [
        carla.Location(x=-ego_ext.x, y=-ego_ext.y, z=0),
        carla.Location(x=-ego_ext.x, y= ego_ext.y, z=0),
        carla.Location(x= ego_ext.x, y=-ego_ext.y, z=0),
        carla.Location(x= ego_ext.x, y= ego_ext.y, z=0)
    ]
    ego_p = [np.array([ego_trans.transform(c).x, ego_trans.transform(c).y]) for c in ego_corners_local]
    
    # Door segments in world coordinates (assuming front doors FL and FR)
    door_trans = door_vehicle.get_transform()
    door_ext = door_vehicle.bounding_box.extent
    # Approx hinge at x=0.5*ext.x, extending 1m laterally
    door_segments = []
    for side in [-1, 1]: # -1 for Left (FL), 1 for Right (FR)
        h_loc = carla.Location(x=0.5*door_ext.x, y=side*door_ext.y, z=0)
        e_loc = carla.Location(x=0.5*door_ext.x, y=side*(door_ext.y + 1.0), z=0)
        h_world = door_trans.transform(h_loc)
        e_world = door_trans.transform(e_loc)
        door_segments.append((np.array([h_world.x, h_world.y]), np.array([e_world.x, e_world.y])))
        
    # Check if any door endpoint is inside ego bounding box
    ego_yaw_rad = np.deg2rad(ego_trans.rotation.yaw)
    cos_a = np.cos(-ego_yaw_rad)
    sin_a = np.sin(-ego_yaw_rad)
    
    min_dist = float('inf')
    for s1, s2 in door_segments:
        # Check intersection by looking at endpoints in ego-local space
        for pt in [s1, s2]:
            dx = pt[0] - ego_trans.location.x
            dy = pt[1] - ego_trans.location.y
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            if abs(lx) <= ego_ext.x and abs(ly) <= ego_ext.y:
                return 0.0

        # Corner to segment
        for p in ego_p:
            min_dist = min(min_dist, distance_point_to_segment(p, s1, s2))
        # Segment endpoints to ego edges
        # Edges: (0,1), (1,3), (3,2), (2,0)
        ego_edges = [(ego_p[0], ego_p[1]), (ego_p[1], ego_p[3]), (ego_p[3], ego_p[2]), (ego_p[2], ego_p[0])]
        for ep1, ep2 in ego_edges:
            min_dist = min(min_dist, distance_point_to_segment(s1, ep1, ep2))
            min_dist = min(min_dist, distance_point_to_segment(s2, ep1, ep2))
            
    return min_dist

# ---------------------------------------------------------------
# ---------------------------------------------------------------

checkpoint_path = (
    '/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/'
    'vla_adapter/model/checkpoints/epoch=013.ckpt/pytorch_model.pt'
)
# ---------------------------------------------------------------
# ---------------------------------------------------------------


class Mode(Enum):
    NORMAL = 1
    ALTERNATE = 2  # latency compensation


class EventType(Enum):
    PERCEPTION = 1


class Event:
    def __init__(self, type: EventType, time: int, data):
        self.type = type
        self.time = time
        self.data = data
# ---------------------------------------------------------------
# ---------------------------------------------------------------

def run_scenario(
    world,
    destination_parking_spot,
    parked_spots,
    ious,
    collisions_ref,
    near_misses_ref,
    actual_collisions,
    walker_collisions_ref,
    recording_path=None,
    car_list=[],
    mode = None
):
    cam1 = None
    cam2 = None
    adapter = None
    parking_scenario = None
    trajectory_log = []
    scenario = ParkingScenarioMedium
    if mode is None:
        mode = HardMode.DoorMode
        scenario = ParkingScenarioHard
    NEAR_MISS_THRESHOLD = 2.0

    # ---------------------------------------------------------------
    # Recording bookkeeping

    path_chase   = recording_path.replace('.mp4', '_chase.mp4')   if recording_path else None
    path_topdown = recording_path.replace('.mp4', '_topdown.mp4') if recording_path else None
    # ---------------------------------------------------------------

    try:
        # ---------------------------------------------------------------
        ## Set up Carla Scenario 
        config = ScenarioConfiguration()
        config.name = "Parking"
        config.type = "Parking"
        config.town = "Town04_Opt"
        config.other_actors = None
        config.route = False
        parking_scenario = scenario(
            world=world,
            config=config,
            destination=destination_parking_spot,
            parked=parked_spots,
            debug_mode=0,
            criteria_enable=True,
            mode=mode,
        )
        # ---------------------------------------------------------------
        # More recording bookkeeping
        if recording_path:
            cam1 = init_recording(parking_scenario.car, path_chase, top_down=False)
            cam2 = init_recording(parking_scenario.car, path_topdown, top_down=True)
        # ---------------------------------------------------------------

        world.tick()
        adapter = DiffusionAdapter()

    
        # ---------------------------------------------------------------
        ## Make destination more extreme to guide the model
        # Compute far-wall destination: where the front bumper should end up.
        # is_done uses the same far-wall point, so the check fires exactly when the car arrives.
        dest_raw = parking_scenario.car.car.destination
        half_len = parking_scenario.car.actor.bounding_box.extent.x
        front_len = half_len + 1.6
        far_dest = TrajectoryPoint(
            dest_raw.direction,
            dest_raw.x + front_len * np.cos(dest_raw.angle),
            dest_raw.y + front_len * np.sin(dest_raw.angle),
            dest_raw.speed,
            dest_raw.angle,
        )
        # world.debug.draw_point(
        #     carla.Location(x=far_dest.x, y=far_dest.y, z=0.5),
        #     size=0.1,
        #     color=carla.Color(r=0, g=200, b=255),
        #     life_time=30,
        # )

        # ---------------------------------------------------------------
        # Initialize model adapter
        adapter.init_testbed(
            checkpoint_path,
            world,
            parking_scenario.car.actor,
            far_dest,
            parking_scenario.car.car.destination.angle,
        )

        # ---------------------------------------------------------------
        # ---------------------------------------------------------------
        ### Give the car access to obstacle bounding boxes (not needed for the model)

        moving_cars_bbs, traffic_cone_bbs, walker_bbs = get_bounding_boxes(parking_scenario)
        all_bbs_list = (
            parking_scenario.parked_cars_bbs
            + traffic_cone_bbs
            + moving_cars_bbs
            + [bb for _, bb in walker_bbs]
        )
        # Use only parked car bbs for the car's static obs. 
        ego_loc = parking_scenario.car.actor.get_location()
        parking_scenario.car.car.obs = obstacle_map_from_bbs(
            parking_scenario.parked_cars_bbs, min_y=ego_loc.y - 5,
        )
        car_list[0] = parking_scenario.car
        parking_scenario.car.car.obs.parked_cars = parking_scenario.parked_cars
        collision_tracker = obstacle_map_from_bbs(all_bbs_list)

        # Pre-compute static cone obstacle map — cones don't move so there's no
        # need to rebuild this grid every tick inside track_collisions().
        if traffic_cone_bbs:
            _co = obstacle_map_from_bbs(traffic_cone_bbs, collision_tracker).obs
            _co[0, :] = 0; _co[-1, :] = 0; _co[:, 0] = 0; _co[:, -1] = 0
            cone_obs_cached = _co | np.roll(_co, 1, 0) | np.roll(_co, -1, 0) | np.roll(_co, 1, 1) | np.roll(_co, -1, 1)
        else:
            cone_obs_cached = None

        # ---------------------------------------------------------------
        # ---------------------------------------------------------------

        ### Initialize Collision Tracking Logic
        vehicle_criterion = next(
            (c for c in parking_scenario.get_criteria() if c.name == "VehicleCollisionTest"), None
        )
        prev_vehicle_count = 0
        colliding_walker_ids = set()   # IDs counted as currently in collision
        walker_last_seen = {}          # {id: ticks_since_last_overlap} for hysteresis
        near_miss_active = False
        near_miss_min_distance = float('inf')
        # ---------------------------------------------------------------

        world.tick()

        # ---------------------------------------------------------------
        scenario_start_wall = time.time()
        skip_requested = False
        original_sigint = signal.getsignal(signal.SIGINT)
        def _skip_handler(signum, frame):
            nonlocal skip_requested
            skip_requested = True
            print("\n[SKIP] Ctrl+C pressed — finishing current scenario and moving to next...")
        signal.signal(signal.SIGINT, _skip_handler)
        # ---------------------------------------------------------------
        # ---------------------------------------------------------------

        while not skip_requested and not adapter.is_done(parking_scenario.car.car.destination):
            world.tick()
            timestamp = world.get_snapshot().timestamp
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()  # needed so that trigger distance will work
            parking_scenario.car.car.localize()

            control = adapter.run_step_testbed(timestamp)
            print(f"applying: t={control.throttle:.3f} s={control.steer:.3f} b={control.brake:.3f}")
            parking_scenario.car.actor.apply_control(control)

            if adapter.latest_pred_route is not None:
                loc = parking_scenario.car.actor.get_location()
                yaw = np.deg2rad(parking_scenario.car.actor.get_transform().rotation.yaw)
                world_traj = ego_to_world(adapter.latest_pred_route, loc.x, loc.y, yaw)
                trajectory_log.append(world_traj)
                parking_scenario.car.latest_trajectory = world_traj

            if parking_scenario.scenario_tree:
                parking_scenario.scenario_tree.tick_once()

            # ---------------------------------------------------------------
            # --- Collision logic ---
            moving_cars_bbs, traffic_cone_bbs, walker_bbs = get_bounding_boxes(parking_scenario)

            prev_vehicle_count, near_miss_active, near_miss_min_distance = track_collisions(
                parking_scenario,
                collision_tracker,
                traffic_cone_bbs,
                walker_bbs,
                vehicle_criterion,
                prev_vehicle_count,
                colliding_walker_ids,
                walker_last_seen,
                near_miss_active,
                near_miss_min_distance,
                collisions_ref,
                walker_collisions_ref,
                near_misses_ref,
                NEAR_MISS_THRESHOLD,
                cone_obs_cached=cone_obs_cached,
            )
            # ---------------------------------------------------------------

            if recording_path:
                process_recording_frames(parking_scenario.car)

            if parking_scenario.scenario_tree.status != py_trees.common.Status.RUNNING:
                print(f"Parking Timeout: Scenario Failure")
                break

            if parking_scenario.timeout_node and parking_scenario.timeout_node.timeout or time.time() - scenario_start_wall > WALL_TIMEOUT:
                print(f"Parking Timeout: 60s elapsed")
                break

    finally:
        signal.signal(signal.SIGINT, original_sigint)
        iou = parking_scenario.car.iou()
        ious.append(iou)
        print(f'IOU: {iou}')
        print(f'Vehicle Collisions (CARLA sensor): {collisions_ref[0]}')
        print(f'Walker/Cone Collisions (BB): {walker_collisions_ref[0]}')
        actual_collisions.append(analyze_scenario(parking_scenario)[1])
        if recording_path and parking_scenario:
            finalize_recording(parking_scenario.car)
        if adapter:
            adapter.destroy_cam()
        if cam1:
            cam1.destroy()
        if cam2:
            cam2.destroy()
        if parking_scenario:
            parking_scenario.cleanup()

        world.tick()


## Inspired by ScenarioManager
def analyze_scenario(parking_scenario):
    """
    This function is intended to be called from outside and provide
    the final statistics about the scenario (human-readable, in form of a junit
    report, etc.)
    """
    result = "SUCCESS"
    print("EVENTS:")
    num_collisions = 0
    for criterion in parking_scenario.get_criteria():
        criterion.terminate(None)
        if hasattr(criterion, 'actual_value'):
            num_collisions += criterion.actual_value
        print(criterion.name, criterion.test_status)
        if criterion.test_status != "SUCCESS":
            result = "SCENARIO FAILURE"

    print("FINAL SCENARIO RESULT:", result)
    print("ACTUAL NUMBER OF COLLISIONS:", num_collisions)
    return result, num_collisions


def make_run_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = os.path.join(script_dir, 'results', timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Generate guitar tablature for audio.")
    parser.add_argument("--mode", type=lambda x: CollisionMode[x], required=False, default=None, help="Scenario Mode")
    args = parser.parse_args()
    run_dir = make_run_dir()
    all_recording_paths = []
    car_list = [None]

    handler = make_cleanup_handler(lambda: car_list[0], all_recording_paths)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    print(f'Saving results to: {run_dir}')

    try:
        client = load_client()

        world = town04_load(client)

        CarlaDataProvider.set_client(client)
        CarlaDataProvider.set_world(world)

        town04_spectator_bev(world)

        ious = []
        collisions = []
        walker_collisions = []
        near_misses = []
        actual_collisions = []
        for i, (destination_parking_spot, parked_spots) in enumerate(SCENARIOS):
            print(f"Scenario {i+1}/{len(SCENARIOS)}")
            print(f'Running scenario: destination={destination_parking_spot}, parked_spots={parked_spots}')
            collisions_ref, near_misses_ref, walker_collisions_ref = [0], [0], [0]
            recording_path = None
            if RECORD:
                recording_path = os.path.join(run_dir, f'scenario_{i+1}.mp4')
                all_recording_paths.append(recording_path.replace('.mp4', '_chase.mp4'))
                all_recording_paths.append(recording_path.replace('.mp4', '_topdown.mp4'))
            run_scenario(
                world,
                destination_parking_spot,
                parked_spots,
                ious,
                collisions_ref,
                near_misses_ref,
                actual_collisions,
                walker_collisions_ref,
                recording_path,
                car_list,
                mode= args.mode
            )
            collisions.append(collisions_ref[0])
            walker_collisions.append(walker_collisions_ref[0])
            near_misses.append(near_misses_ref[0])

    except KeyboardInterrupt:
        print('stopping simulation')

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    finally:
        summary = {
            'vehicle_collisions': collisions,
            'total_vehicle_collisions': sum(collisions) if collisions else 0,
            'vehicle_collision_rate': sum(collisions) / len(collisions) if collisions else 0,
            'walker_cone_collisions': walker_collisions,
            'total_walker_cone_collisions': sum(walker_collisions) if walker_collisions else 0,
            'walker_cone_collision_rate': sum(walker_collisions) / len(walker_collisions) if walker_collisions else 0,
            'actual_collisions': actual_collisions,
            'actual_total_collisions': sum(actual_collisions) if actual_collisions else 0,
            'actual_collision_rate': sum(actual_collisions) / len(actual_collisions) if actual_collisions else 0,
            'near_misses': near_misses,
            'total_near_misses': sum(near_misses) if near_misses else 0,
            'ious': ious,
            'mean_iou': float(np.mean(ious)) if ious else 0,
            'median_iou': float(np.median(ious)) if ious else 0,
        }
        with open(os.path.join(run_dir, 'results.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'Saved results.json')

        if collisions:
            print(f"\nVehicle Collision Statistics (CARLA sensor):")
            print(f"  Total vehicle collisions: {sum(collisions)}")
            print(f"  Collision counts: {collisions}")
            print(f"  Collision Rate (per scenario): {sum(collisions)/len(collisions):.3f}")
        if walker_collisions:
            print(f"\nWalker/Cone Collision Statistics (BB):")
            print(f"  Total walker/cone collisions: {sum(walker_collisions)}")
            print(f"  Collision counts: {walker_collisions}")
            print(f"  Collision Rate (per scenario): {sum(walker_collisions)/len(walker_collisions):.3f}")
        if actual_collisions:
            print(f"\nActual Collision Statistics:")
            print(f"  Actual Total collisions: {sum(actual_collisions)}")
            print(f"  Actual Collision counts: {actual_collisions}")
            print(f"  Actual Collision Rate (per scenario): {sum(actual_collisions)/len(actual_collisions):.3f}")
        if ious:
            print("Final IOUS:")
            print(ious)
            plt.clf()
            plt.boxplot(ious, positions=[1], vert=True, patch_artist=True, widths=0.5,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red'))
            x_scatter = np.random.normal(loc=0.5, scale=0.05, size=len(ious))
            plt.scatter(x_scatter, ious, color='darkblue', alpha=0.6, label='Data Points')
            plt.xticks([1], ['IOU Values'])
            plt.title('Parking IOU Values')
            plt.ylabel('IOU Value')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            plt.savefig(os.path.join(run_dir, 'iou_boxplot.png'))
            print(f'Saved iou_boxplot.png')

        world.tick()
        print(f'\nAll results saved to: {run_dir}')
    # ---------------------------------------------------------------
    # ---------------------------------------------------------------

def track_collisions(
    parking_scenario,
    collision_tracker,
    traffic_cone_bbs,
    walker_bbs,
    vehicle_criterion,
    prev_vehicle_count,
    colliding_walker_ids,
    walker_last_seen,
    near_miss_active,
    near_miss_min_distance,
    collisions_ref,
    walker_collisions_ref,
    near_misses_ref,
    near_miss_threshold,
    cone_obs_cached=None,
):
    """
    Run one tick of collision and near-miss tracking. Mutates colliding_walker_ids and
    walker_last_seen in-place; returns updated scalar state as a tuple.
    """
    # Vehicle collision: read CollisionTest criterion (same source as actual_collisions)
    if vehicle_criterion is not None:
        new_count = vehicle_criterion.actual_value
        if new_count > prev_vehicle_count:
            collisions_ref[0] += new_count - prev_vehicle_count
            print(f"Vehicle Collision Detected! (total so far: {new_count})")
            print("Current car location", parking_scenario.car.actor.get_location())
        prev_vehicle_count = new_count

    # Walker / cone collision + near-miss: grid-based, per-actor debounce
    car = parking_scenario.car.car
    collision_mask = collision_tracker.generate_collision_mask(
        car.cur, front_m=car.front_m, rear_m=car.rear_m, half_width_m=car.half_width_m,
    )

    # Per-walker check: exact OBB-AABB SAT — no grid quantization or dilation errors.
    walker_ids_this_tick = set()
    for w_id, w_bb in walker_bbs:
        if obb_aabb_overlap(car.cur, car.front_m, car.rear_m, car.half_width_m, w_bb):
            walker_ids_this_tick.add(w_id)

    # Cones: no per-actor IDs, use sentinel key 'cone'.
    # Use pre-computed dilated map when available (cones are static).
    if cone_obs_cached is not None:
        if np.any(collision_mask & (cone_obs_cached == 1)):
            walker_ids_this_tick.add('cone')
    elif traffic_cone_bbs:
        cone_obs = obstacle_map_from_bbs(traffic_cone_bbs, collision_tracker).obs
        cone_obs[0, :] = 0; cone_obs[-1, :] = 0
        cone_obs[:, 0] = 0; cone_obs[:, -1] = 0
        _c = cone_obs
        cone_obs = _c | np.roll(_c, 1, 0) | np.roll(_c, -1, 0) | np.roll(_c, 1, 1) | np.roll(_c, -1, 1)
        if np.any(collision_mask & (cone_obs == 1)):
            walker_ids_this_tick.add('cone')

    # Hysteresis: keep an ID in the active set for up to GRACE_TICKS after
    # the last detected overlap, so grid-quantization flicker doesn't re-count
    # the same walker crossing event.
    GRACE_TICKS = 20  # ~1 s at 20 Hz; a walker can't re-enter that fast
    for wid in walker_ids_this_tick:
        walker_last_seen[wid] = 0  # reset / start grace counter
    for wid in list(walker_last_seen.keys()):
        if wid not in walker_ids_this_tick:
            walker_last_seen[wid] += 1
            if walker_last_seen[wid] > GRACE_TICKS:
                colliding_walker_ids.discard(wid)
                del walker_last_seen[wid]

    new_collisions = walker_ids_this_tick - colliding_walker_ids
    if new_collisions:
        walker_collisions_ref[0] += len(new_collisions)
        print(f"Walker/Cone Collision(s) Detected: {new_collisions}")
    colliding_walker_ids |= walker_ids_this_tick

    # Near-miss (only when no active collision)
    all_walker_bbs = [bb for _, bb in walker_bbs] + traffic_cone_bbs
    if not walker_ids_this_tick and all_walker_bbs:
        combined_obs = obstacle_map_from_bbs(all_walker_bbs, collision_tracker).obs
        combined_obs[0, :] = 0; combined_obs[-1, :] = 0
        combined_obs[:, 0] = 0; combined_obs[:, -1] = 0
        _w2 = combined_obs
        combined_obs = _w2 | np.roll(_w2, 1, 0) | np.roll(_w2, -1, 0) | np.roll(_w2, 1, 1) | np.roll(_w2, -1, 1)
        distance = near_miss(collision_mask, combined_obs)
        if 0 < distance < near_miss_threshold:
            near_miss_min_distance = min(near_miss_min_distance, distance) if near_miss_active else distance
            near_miss_active = True
        else:
            if near_miss_active:
                near_misses_ref[0] += 1
                print(f"Near-miss! Closest distance: {near_miss_min_distance:.2f}m")
            near_miss_active = False
    elif not walker_ids_this_tick:
        near_miss_active = False

    return prev_vehicle_count, near_miss_active, near_miss_min_distance



def obb_aabb_overlap(cur, front_m, rear_m, half_w, aabb):
    """Exact OBB-AABB overlap via Separating Axis Theorem. No grid, no quantization."""
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


def draw_car(parking_scenario, world):
    car_transform = parking_scenario.car.actor.get_transform()
    loc = car_transform.location

    car = parking_scenario.car.car
    length_front = car.front_m
    length_rear  = car.rear_m
    width        = car.half_width_m * 2
    cur = car.cur
    yaw = cur.angle  # already in radians

    local_corners = np.array([
        [-length_rear, -width/2],
        [-length_rear,  width/2],
        [ length_front,  width/2],
        [ length_front, -width/2],
    ])
    R = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)],
    ])

    world_corners = [np.dot(R, corner) + np.array([cur.x, cur.y]) for corner in local_corners]

    for i in range(4):
        p1 = world_corners[i]
        p2 = world_corners[(i+1) % 4]
        world.debug.draw_line(
            carla.Location(x=p1[0], y=p1[1], z=loc.z + 0.1),
            carla.Location(x=p2[0], y=p2[1], z=loc.z + 0.1),
            thickness=0.5,
            color=carla.Color(r=255, g=0, b=0),
            life_time=1,
            persistent_lines=True,
        )


if __name__ == '__main__':
    
    main()
