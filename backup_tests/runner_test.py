from srunner.scenariomanager.actorcontrols.pedestrian_control import PedestrianControl
from v2_experiment_utils import (
    load_client,
    is_done,
    town04_load,
    town04_spectator_bev,
    town04_spawn_ego_vehicle,
    town04_spawn_parked_cars,
    town04_spectator_follow,
    town04_get_drivable_graph
)

# For lane waypoint hack
from parking_position import (
    parking_lane_waypoints_Town04
)

import carla

import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio

from srunner.scenariomanager.actorcontrols.basic_control import BasicControl
import random

from v2_experiment import SCENARIOS
NUM_RANDOM_CARS = 25

def spawn_pedestrian(world, start_location, end_location, velocity = 1.5):
    # inspired by pedestrian_crossing scenario

    blueprint_library = world.get_blueprint_library()
    pedestrian_bp = blueprint_library.filter('walker.pedestrian.*')[0]

    spawn_point = carla.Transform(start_location, carla.Rotation(yaw=90))
    pedestrian = world.spawn_actor(pedestrian_bp, spawn_point)

    pedestrian.set_simulate_physics(True)

    # pedestrian.set_target_velocity(velocity)

    controller = PedestrianControl(pedestrian)
    end_waypoint = carla.Transform(
        carla.Location(x=end_location.x, y=end_location.y, z=end_location.z),
        carla.Rotation()
    )

    controller.update_target_speed(1.5)
    controller.update_waypoints([end_waypoint]) 

    return pedestrian, controller

def run_scenario(world, destination_parking_spot, parked_spots, ious, recording_file):
    pedestrians = []
    pedestrian_controllers = []
    try:
        # load parked cars
        parked_cars, parked_cars_bbs = town04_spawn_parked_cars(world, parked_spots, destination_parking_spot, NUM_RANDOM_CARS)

        # load car
        car = town04_spawn_ego_vehicle(world, destination_parking_spot)
        
        world.tick()
        world.tick()

        recording_cam = car.init_recording(recording_file)

        # HACK: enable perfect perception of parked cars
        car.car.obs = parked_cars_bbs

        # HACK: set lane waypoints to guide parking in adjacent lanes
        car.car.lane_wps = parking_lane_waypoints_Town04

        # pedestrian, ped_controller = spawn_pedestrian(world, ped_start_location, ped_end_location)
        # pedestrians.append(pedestrian)
    
        
        # DEBUG: Draw marker
        # world.debug.draw_point(start_location, size=0.5, color=carla.Color(255, 0, 0), life_time=120.0)
        # world.debug.draw_string(start_location, 'PEDESTRIAN', color=carla.Color(255, 0, 0), life_time=120.0)
        num_pedestrians = random.randint(4,15)
        spawn_times = [random.randint(0, 500) for _ in range(num_pedestrians)]
        ped_controller = None
        # run simulation
        i = 0
        while not is_done(car):
            world.tick()
            car.run_step()
            
            car.process_recording_frames()
            if (i in spawn_times):
                ego_location = car.actor.get_location()
                ped_start_location = carla.Location(ego_location.x - 5, ego_location.y + 10, ego_location.z+1)
                ped_end_location = carla.Location(ego_location.x + 2, ego_location.y + 10, ego_location.z+1)
                pedestrian, ped_controller = spawn_pedestrian(world, ped_start_location, ped_end_location)
                pedestrians.append(pedestrian)
                pedestrian_controllers.append(ped_controller)
            for controller in pedestrian_controllers:
                controller.run_step()
            i += 1
            
            # town04_spectator_follow(world, car)

        iou = car.iou()
        ious.append(iou)
        print(f'IOU: {iou}')
    finally:
        recording_cam.destroy()
        car.destroy()

        for pedestrian in pedestrians:
            if pedestrian.is_alive:
                pedestrian.destroy()

        for parked_car in parked_cars:
            parked_car.destroy()

def main():
    try:
        client = load_client()

        # load map
        world = town04_load(client)

        # load spectator
        town04_spectator_bev(world)

        # load recording file
        recording_file = iio.imopen('./test.mp4', 'w', plugin='pyav')
        recording_file.init_video_stream('vp9', fps=30)

        # run scenarios
        ious = []
        for destination_parking_spot, parked_spots in SCENARIOS:
            print(f'Running scenario: destination={destination_parking_spot}, parked_spots={parked_spots}')
            run_scenario(world, destination_parking_spot, parked_spots, ious, recording_file)

        # graph ious
        plt.clf()
        plt.boxplot(ious, positions=[1], vert=True, patch_artist=True, widths=0.5,
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'))
        x_scatter = np.random.normal(loc=0.5, scale=0.05, size=len(ious))
        plt.scatter(x_scatter, ious, color='darkblue', alpha=0.6, label='Data Points')
        plt.xticks([1], ['IOU Values'])  # Set x-ticks at the boxplot
        plt.title('Parking IOU Values')
        plt.ylabel('IOU Value')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.savefig('iou_boxplot.png')

    except KeyboardInterrupt:
        print('stopping simulation')
    
    finally:
        recording_file.close()
        world.tick()

if __name__ == '__main__':
    main()