"""benchmark.py — benchmark runner mirroring runner_test_medium.py.

Switch planners by setting PLANNER below:
    PLANNER = 'hybrid_a_star'   — uses v2.py's hybrid A* (validates infrastructure)
    PLANNER = 'diffusion'       — uses diffusion_adapter (validates diffusion wiring)

Results (per-scenario .mp4 + results.json) are saved to results/<planner>_<timestamp>/.

Run from the autovalet directory:
    cd leaderboard/leaderboard/autovalet
    python diffusion_adapter/benchmark.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from enum import Enum

import numpy as np

# Allow imports from parent autovalet directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import carla
import py_trees

from parking_scenarios.parking_scenario_hard import HardMode, ParkingScenarioHard
from parking_scenarios.opposite_vehicle_parking import CollisionMode
from v2 import ObstacleMap
from srunner.scenariomanager.timer import GameTime
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from v2_experiment_utils import (
    _draw_bb,
    clear_obstacle_map,
    get_bounding_boxes,
    load_client,
    is_done,
    near_miss,
    obstacle_map_from_bbs,
    town04_load,
    town04_spectator_bev,
)
from parking_position import (
    parking_lane_waypoints_Town04,
    parking_vehicle_locations_Town04,
)
from v2_experiment import SCENARIOS
import matplotlib.pyplot as plt

from diffusion_adapter import v2_diffusion
from diffusion_adapter.v2_diffusion import CarlaCar as DiffusionCarlaCar

NUM_RANDOM_CARS = 1
NETWORK_SEND_LATENCIES = [0] # ms
PERCEPTION_LATENCY = 200 # ms
SMALL_PERCEPTION_LATENCY = 100 # ms
RECV_LATENCY = 100 # ms
MODE_PERIOD = 500 # ms
PLANNING_PERIOD = 500 # ms
DATA_COLLECTION_PERIOD = 200 # ms
SHOULD_PIPELINE = True
SHOULD_ADJUST_MODEL = False
TIMEOUT = 120 * 1000 # ms
IMAGE_DOWNSIZE = 1
WALL_TIMEOUT = 120 # s
RECORD = True
RECORD_TOP_DOWN = True


NETWORK_SEND_LATENCIES = [latency // IMAGE_DOWNSIZE for latency in NETWORK_SEND_LATENCIES]

import time

PLANNER = "diffusion"

class Mode(Enum):
    NORMAL = 1
    ALTERNATE = 2 # latency compensation

class EventType(Enum):
    PERCEPTION = 1

class Event:
    def __init__(self, type: EventType, time: int, data):
        self.type = type
        self.time = time
        self.data = data

def run_scenario(world, destination_parking_spot, parked_spots, ious, collisions, collisions_ref, near_misses, near_misses_ref, actual_collisions, walker_collisions, walker_collisions_ref, recording_path=None):
    recording_cam = None
    parking_scenario = None
    mode = CollisionMode.MISS ### TODO: make this an argument
    NEAR_MISS_THRESHOLD = 2.0
    PREDICTION_HORIZON = 0.5

    try:  

        config = ScenarioConfiguration()
        config.name = "Parking"
        config.type = "Parking"
        config.town = "Town04_Opt"
        config.other_actors = None
        config.route = False

        ### Load the scenario
        parking_scenario = ParkingScenarioHard(
            world=world,
            config=config,
            destination=destination_parking_spot,
            parked=parked_spots,
            debug_mode=0,
            criteria_enable=True,
            mode=HardMode.PedMode,
            car_class=DiffusionCarlaCar,
        )

        world.tick()
        world.tick()

        if recording_path:
            recording_cam = parking_scenario.car.init_recording(recording_path, top_down=RECORD_TOP_DOWN)

        #### SUPER IMPORTANT: something with build scenarios breaks the car's motions
        ###### There are three possible fixes, 
        # to call small run_step here (bad because we would have to set obstacles BEFORE scenario)
        # to set a min_speed
        

        moving_cars_bbs, traffic_cone_bbs, walker_bbs = get_bounding_boxes(parking_scenario)
        all_bbs_list = parking_scenario.parked_cars_bbs + traffic_cone_bbs + moving_cars_bbs + [bb for _, bb in walker_bbs]
        
        # Use only parked car bbs for the car's static obs. Walkers/moving obstacles
        # are dynamic and tracked by ground_truth_kalman_filter; baking them into
        # static_obs at their initial positions creates ghost obstacles and corrupts
        # the map when they move (static_obs cells never restored after KF zeroes them).
        ego_loc = parking_scenario.car.actor.get_location()
        parking_scenario.car.car.obs = obstacle_map_from_bbs(
            parking_scenario.parked_cars_bbs, min_y=ego_loc.y - 5)

        # For diffusion Model input    
        parking_scenario.car.car.obs.parked_cars = parking_scenario.parked_cars
        ###
        
        parked_car_ids = {car.id for car in parking_scenario.parked_cars}
        collision_tracker = obstacle_map_from_bbs(all_bbs_list)

        if PLANNER == 'diffusion':
            from diffusion_adapter.adapter import init_scenario
            parking_scenario.car.car.localize()
            init_scenario(
                parking_scenario.car.car.cur,
                parking_scenario.car.car.destination,
                parking_scenario.car.car.obs,
            )


        vehicle_criterion = next(
            (c for c in parking_scenario.get_criteria() if c.name == "VehicleCollisionTest"), None
        )
        prev_vehicle_count = 0
        colliding_walker_ids = set()   # IDs counted as currently in collision
        walker_last_seen = {}          # {id: ticks_since_last_overlap} for hysteresis
        near_miss_active = False
        near_miss_min_distance = float('inf')
        kalman_filters = {}

        world.tick()

        scenario_start_wall = time.time()

        while not is_done(parking_scenario.car):

            ### ----------- Step actors

            world.tick()
            timestamp = world.get_snapshot().timestamp
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick() ### NEED THIS so that trigger distance will work
            parking_scenario.car.car.localize()
            parking_scenario.car.run_step(parked_car_ids)

            if parking_scenario.scenario_tree:
                parking_scenario.scenario_tree.tick_once()


            #--------- Collision Logic

            draw_car(parking_scenario, world)
            moving_cars_bbs, traffic_cone_bbs, walker_bbs = get_bounding_boxes(parking_scenario)

            for bb in parking_scenario.parked_cars_bbs:
                _draw_bb(world, bb, carla.Color(0, 0, 255), 1)

            # --- Vehicle collision: read CollisionTest criterion (same source as actual_collisions) ---
            if vehicle_criterion is not None:
                new_count = vehicle_criterion.actual_value
                if new_count > prev_vehicle_count:
                    collisions_ref[0] += new_count - prev_vehicle_count
                    print(f"Vehicle Collision Detected! (total so far: {new_count})")
                    print("Current car location", parking_scenario.car.actor.get_location())
                prev_vehicle_count = new_count

            # --- Walker / cone collision + near-miss: grid-based, per-actor debounce ---
            car = parking_scenario.car.car
            collision_mask = collision_tracker.generate_collision_mask(
                car.cur, front_m=car.front_m, rear_m=car.rear_m, half_width_m=car.half_width_m,
            )

            # Per-walker check: exact OBB-AABB SAT — no grid quantization or dilation errors.
            walker_ids_this_tick = set()
            for w_id, w_bb in walker_bbs:
                if obb_aabb_overlap(car.cur, car.front_m, car.rear_m, car.half_width_m, w_bb):
                    walker_ids_this_tick.add(w_id)

            # Cones: no per-actor IDs, use sentinel key 'cone'
            if traffic_cone_bbs:
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
                if 0 < distance < NEAR_MISS_THRESHOLD:
                    near_miss_min_distance = min(near_miss_min_distance, distance) if near_miss_active else distance
                    near_miss_active = True
                else:
                    if near_miss_active:
                        near_misses_ref[0] += 1
                        print(f"Near-miss! Closest distance: {near_miss_min_distance:.2f}m")
                    near_miss_active = False
            elif not walker_ids_this_tick:
                near_miss_active = False

            #--------

            if recording_path:
                parking_scenario.car.process_recording_frames()

            if parking_scenario.scenario_tree.status != py_trees.common.Status.RUNNING:
                print(f"Parking Timeout: Scenario Failure")
                break

            if time.time() - scenario_start_wall > WALL_TIMEOUT:
                print(f"Parking Timeout: 60s elapsed")
                break

            
            
        
        # iou = parking_scenario.car.iou()
        # ious.append(iou)
        # print(f'IOU: {iou}')
        print(f'Vehicle Collisions (CARLA sensor): {collisions_ref[0]}')
        print(f'Walker/Cone Collisions (BB): {walker_collisions_ref[0]}')
    finally:
        iou = parking_scenario.car.iou()
        ious.append(iou)
        print(f'IOU (Final Position): {iou}')
        actual_collisions.append(analyze_scenario(parking_scenario)[1])
        collisions.append(collisions_ref[0])
        walker_collisions.append(walker_collisions_ref[0])
        near_misses.append(near_misses_ref[0])
        if recording_cam:
            recording_cam.destroy()
        if recording_path:
            parking_scenario.car.finalize_recording()
        if parking_scenario:
            parking_scenario.cleanup()

        world.tick()
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
        
        # parking_scenario.scenario_tree.terminate(py_trees.common.Status.SUCCESS)

def make_run_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = os.path.join(script_dir, 'results', timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def main():
    run_dir = make_run_dir()
    print(f'Saving results to: {run_dir}')

    from diffusion_adapter.adapter import load_model
    _dir = os.path.dirname(os.path.abspath(__file__))
    load_model(
        os.path.join(_dir, 'checkpoints', 'model.pth'),
        os.path.join(_dir, 'checkpoints', 'args.json'),
    )

    try:

        client = load_client()

        # load map
        world = town04_load(client)

        CarlaDataProvider.set_client(client)
        CarlaDataProvider.set_world(world)

        # load spectator
        town04_spectator_bev(world)

        # run scenarios
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
            run_scenario(world, destination_parking_spot, parked_spots, ious, collisions, collisions_ref, near_misses, near_misses_ref, actual_collisions, walker_collisions, walker_collisions_ref, recording_path)

    except KeyboardInterrupt:
        print('stopping simulation')

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
    # Use car.car.cur to match exactly what generate_collision_mask uses
    cur = car.cur
    yaw = cur.angle  # already in radians

    # define rectangle corners in local frame
    local_corners = np.array([
        [-length_rear, -width/2],
        [-length_rear,  width/2],
        [ length_front,  width/2],
        [ length_front, -width/2]
    ])
    R = np.array([[np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)]])

    world_corners = [np.dot(R, corner) + np.array([cur.x, cur.y]) for corner in local_corners]

    for i in range(4):
        p1 = world_corners[i]
        p2 = world_corners[(i+1)%4]
        world.debug.draw_line(
            carla.Location(x=p1[0], y=p1[1], z=loc.z + 0.1),
            carla.Location(x=p2[0], y=p2[1], z=loc.z + 0.1),
            thickness=0.5,
            color=carla.Color(r=255, g=0, b=0),
            life_time=1,
            persistent_lines=True
        )


if __name__ == '__main__':
    main()