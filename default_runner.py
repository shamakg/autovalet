### Version 3 of the code, with the goal being to renew the pedestrian crossing abstraction

import argparse
import json
import os
from datetime import datetime
from enum import Enum
# from parking_scenarios.parking_scenario import ParkingScenario
from parking_scenarios.parking_scenario_easy import ParkingScenarioEasy

from parking_scenarios.parking_scenario_medium import ParkingScenarioMedium
from parking_scenarios.parking_scenario_hard import HardMode, ParkingScenarioHard
from parking_scenarios.opposite_vehicle_parking import CollisionMode
from vla_adapter.v2 import ObstacleMap
from srunner.scenariomanager.timer import GameTime
from srunner.scenarioconfigs.scenario_configuration import ActorConfigurationData, ScenarioConfiguration
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

import py_trees
from testbed.v2_experiment_utils import (
    _draw_bb,
    clear_obstacle_map,
    get_bounding_boxes,

    load_client,
    is_done,
    obstacle_map_from_bbs,
    town04_load,
    town04_spawn_parked_cars,
    town04_spawn_parked_cars_with_doors,
    town04_spectator_bev,

)

from testbed.recording_utils import (
    init_recording,
    process_recording_frames,
    finalize_recording,
    prompt_delete,
    make_cleanup_handler,
    open_recordings

)

# For lane waypoint hack
from parking_position import (
    parking_lane_waypoints_Town04,
    parking_vehicle_locations_Town04,
)

import carla

import numpy as np
import matplotlib.pyplot as plt
import sys
import signal


from v2_experiment import SCENARIOS
from scenario_utils import (
    analyze_scenario, draw_car, track_collisions,
    calculate_min_distance_to_door, compute_weighted_iou,
)
NUM_RANDOM_CARS = 50
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


NETWORK_SEND_LATENCIES = [latency // IMAGE_DOWNSIZE for latency in NETWORK_SEND_LATENCIES]


import time



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

class ScenarioMode(Enum):
    COLLIDE    = "collide"
    NEAR_MISS  = "near_miss"
    STOP_EARLY = "stop_early"
    PEDESTRIAN = "pedestrian"
    DOORMODE   = "doormode"
    CONE       = "cone"

def run_scenario(world, destination_parking_spot, parked_spots, ious, weighted_ious, collisions_ref, near_misses_ref, actual_collisions, walker_collisions_ref, recording_path=None, car_list=[], scenario_mode=ScenarioMode.CONE):
    cam1 = None
    cam2 = None
    parking_scenario = None
    min_door_dist = float('inf')
    door_actor = None
    NEAR_MISS_THRESHOLD = 2.0
    PREDICTION_HORIZON = 0.5

    path_chase   = recording_path.replace('.mp4', '_chase.mp4')   if recording_path else None
    path_topdown = recording_path.replace('.mp4', '_topdown.mp4') if recording_path else None

    try:

        config = ScenarioConfiguration()
        config.name = "Parking"
        config.type = "Parking"
        config.town = "Town04_Opt"
        config.other_actors = None
        config.route = False

        ### Load the scenario based on selected mode
        common_kwargs = dict(
            world=world,
            config=config,
            destination=destination_parking_spot,
            parked=parked_spots,
            debug_mode=0,
            criteria_enable=True,
        )
        if scenario_mode == ScenarioMode.CONE:
            parking_scenario = ParkingScenarioEasy(**common_kwargs)
        elif scenario_mode in (ScenarioMode.COLLIDE, ScenarioMode.NEAR_MISS, ScenarioMode.STOP_EARLY):
            collision_mode_map = {
                ScenarioMode.COLLIDE:    CollisionMode.COLLIDE,
                ScenarioMode.NEAR_MISS:  CollisionMode.MISS,
                ScenarioMode.STOP_EARLY: CollisionMode.STOP_EARLY,
            }
            parking_scenario = ParkingScenarioMedium(**common_kwargs, mode=collision_mode_map[scenario_mode])
        elif scenario_mode == ScenarioMode.PEDESTRIAN:
            parking_scenario = ParkingScenarioHard(**common_kwargs, mode=HardMode.PedMode)
        elif scenario_mode == ScenarioMode.DOORMODE:
            parking_scenario = ParkingScenarioHard(**common_kwargs, mode=HardMode.DoorMode)

        if scenario_mode == ScenarioMode.DOORMODE:
            for s in parking_scenario.list_scenarios:
                if "VehicleOpensDoor" in s.__class__.__name__:
                    door_actor = getattr(s, '_parked_actor', None)
                    break

        if recording_path:
            cam1 = init_recording(parking_scenario.car, path_chase, top_down=False)
            cam2 = init_recording(parking_scenario.car, path_topdown, top_down=True)

        world.tick()
        world.tick()

        

        #### SUPER IMPORTANT: something with build scenarios breaks the car's motions
        ###### There are three possible fixes, 
        # to call small run_step here (bad because we would have to set obstacles BEFORE scenario)
        # to set a min_speed greater than zero
        #### ISSUE ABOVE FIXED
        

        moving_cars_bbs, traffic_cone_bbs, walker_bbs = get_bounding_boxes(parking_scenario)
        all_bbs_list = parking_scenario.parked_cars_bbs + traffic_cone_bbs + moving_cars_bbs + [bb for _, bb in walker_bbs]

        # Use only parked car bbs for the car's static obs. Walkers/moving obstacles
        # are dynamic and tracked by ground_truth_kalman_filter; baking them into
        # static_obs at their initial positions creates ghost obstacles and corrupts
        # the map when they move (static_obs cells never restored after KF zeroes them).
        ego_loc = parking_scenario.car.actor.get_location()
        parking_scenario.car.car.obs = obstacle_map_from_bbs(
            parking_scenario.parked_cars_bbs, min_y=ego_loc.y - 5)
        car_list[0] = parking_scenario.car
        # Give the diffusion adapter access to parked car actors so it can pass
        # their real positions, yaws, and dimensions to the model as agent features.
        parking_scenario.car.car.obs.parked_cars = parking_scenario.parked_cars
        parked_car_ids = {car.id for car in parking_scenario.parked_cars}
        collision_tracker = obstacle_map_from_bbs(all_bbs_list)

        if traffic_cone_bbs:
            _co = obstacle_map_from_bbs(traffic_cone_bbs, collision_tracker).obs
            _co[0, :] = 0; _co[-1, :] = 0; _co[:, 0] = 0; _co[:, -1] = 0
            cone_obs_cached = _co | np.roll(_co, 1, 0) | np.roll(_co, -1, 0) | np.roll(_co, 1, 1) | np.roll(_co, -1, 1)
        else:
            cone_obs_cached = None

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

            ### ----------- tick all the actors

            world.tick()
            timestamp = world.get_snapshot().timestamp
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick() ### NEED THIS so that trigger distance will work
            parking_scenario.car.car.localize()
            parking_scenario.car.run_step(parked_car_ids)
            # parking_scenario.car.process_recording_frames()


            if parking_scenario.scenario_tree:
                parking_scenario.scenario_tree.tick_once()

            if door_actor is not None:
                d = calculate_min_distance_to_door(parking_scenario.car.actor, door_actor)
                if d < min_door_dist:
                    min_door_dist = d

            #--------- Collision Logic

            # draw_car(parking_scenario, world)
            moving_cars_bbs, traffic_cone_bbs, walker_bbs = get_bounding_boxes(parking_scenario)

            # for bb in parking_scenario.parked_cars_bbs:
            #     _draw_bb(world, bb, carla.Color(0, 0, 255), 1)

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

            if recording_path:
                process_recording_frames(parking_scenario.car)

            if parking_scenario.scenario_tree.status != py_trees.common.Status.RUNNING:
                print(f"Parking Timeout: Scenario Failure")
                break

            if parking_scenario.timeout_node and parking_scenario.timeout_node.timeout or time.time() - scenario_start_wall > WALL_TIMEOUT:
                print(f"Parking Timeout: 60s elapsed")
                break

            
            
        
        iou = parking_scenario.car.iou()
        ious.append(iou)
        weighted_ious.append(compute_weighted_iou(iou, scenario_mode == ScenarioMode.DOORMODE, min_door_dist))
        print(f'IOU: {iou}')
        print(f'Vehicle Collisions (CARLA sensor): {collisions_ref[0]}')
        print(f'Walker/Cone Collisions (BB): {walker_collisions_ref[0]}')
    finally:
        if parking_scenario:
            actual_collisions.append(analyze_scenario(parking_scenario)[1])
        if recording_path and parking_scenario:
            finalize_recording(parking_scenario.car)
        if cam1: cam1.destroy()
        if cam2: cam2.destroy()
        if parking_scenario:
            parking_scenario.cleanup()

        world.tick()
        world.tick()

def make_run_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = os.path.join(script_dir, 'results', timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def main():
    parser = argparse.ArgumentParser(description="Autovalet parking test runner")
    parser.add_argument(
        '--mode',
        type=str,
        default='cone',
        choices=[m.value for m in ScenarioMode],
        help="Scenario mode: collide, near_miss, stop_early, pedestrian, doormode, cone (default: cone)",
    )
    args = parser.parse_args()
    scenario_mode = ScenarioMode(args.mode)
    print(f"Running in mode: {scenario_mode.value}")

    run_dir = make_run_dir()
    all_recording_paths = []
    car_list = [None]

    handler = make_cleanup_handler(lambda: car_list[0], all_recording_paths)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    print(f'Saving results to: {run_dir}')

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
        weighted_ious = []
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
            run_scenario(world, destination_parking_spot, parked_spots, ious, weighted_ious, collisions_ref, near_misses_ref, actual_collisions, walker_collisions_ref, recording_path, car_list, scenario_mode)
            collisions.append(collisions_ref[0])
            walker_collisions.append(walker_collisions_ref[0])
            near_misses.append(near_misses_ref[0])

    except KeyboardInterrupt:
        print('stopping simulation')

    finally:
        open_recordings(all_recording_paths)
        prompt_delete(all_recording_paths)
        
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
            'weighted_ious': weighted_ious,
            'mean_weighted_iou': float(np.mean(weighted_ious)) if weighted_ious else 0,
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


if __name__ == '__main__':
    main()