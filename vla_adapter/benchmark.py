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
from agent_interface import SimLingoAdapter
from scenario_utils import (
    analyze_scenario, track_collisions,
    calculate_min_distance_to_door, compute_weighted_iou,
)

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
    weighted_ious,
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
    min_door_dist = float('inf')
    door_actor = None
    scenario = ParkingScenarioMedium
    if mode is None:
        mode = HardMode.PedMode
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

        # Find door actor for safety metric if in DoorMode
        if mode == HardMode.DoorMode:
            for s in parking_scenario.list_scenarios:
                if "VehicleOpensDoor" in s.__class__.__name__:
                    door_actor = getattr(s, '_parked_actor', None)
                    break
        # ---------------------------------------------------------------
        # More recording bookkeeping
        if recording_path:
            cam1 = init_recording(parking_scenario.car, path_chase, top_down=False)
            cam2 = init_recording(parking_scenario.car, path_topdown, top_down=True)
        # ---------------------------------------------------------------

        world.tick()
        adapter = SimLingoAdapter('localhost', 2000)

    
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

            if door_actor is not None:
                d = calculate_min_distance_to_door(parking_scenario.car.actor, door_actor)
                if d < min_door_dist:
                    min_door_dist = d

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
        
        weighted_iou = compute_weighted_iou(iou, mode == HardMode.DoorMode, min_door_dist)
        
        weighted_ious.append(weighted_iou)
        
        print(f'IOU: {iou}')
        print(f'Vehicle Collisions (CARLA sensor): {collisions_ref[0]}')
        print(f'Walker/Cone Collisions (BB): {walker_collisions_ref[0]}')
        if parking_scenario:
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
            run_scenario(
                world,
                destination_parking_spot,
                parked_spots,
                ious,
                weighted_ious,
                collisions_ref,
                near_misses_ref,
                actual_collisions,
                walker_collisions_ref,
                recording_path,
                car_list,
                mode= args.mode,
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
    # ---------------------------------------------------------------
    # ---------------------------------------------------------------

if __name__ == '__main__':
    
    main()
