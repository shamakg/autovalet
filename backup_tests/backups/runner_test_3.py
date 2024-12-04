### Version 3 of the code, with the goal being to renew the pedestrian crossing abstraction

from srunner.scenarioconfigs.scenario_configuration import ActorConfigurationData, ScenarioConfiguration
from srunner.scenariomanager.actorcontrols.pedestrian_control import PedestrianControl
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import OpenVehicleDoor, UpdateAllActorControls
from srunner.scenarios.pedestrian_crossing import PedestrianCrossing
from srunner.scenarios.pedestrian_crossing_parking import PedestrianCrossingParking
from srunner.scenarios.pedestrian_crossing_minimal import PedestrianCrossingMinimal
from srunner.scenarios.vehicle_opens_door import VehicleOpensDoorTwoWays
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

from parking_position import (
    parking_vehicle_locations_Town04,
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

def pedestrian_scenario(world, car, ego_location, num_pedestrians=3):
    config = ScenarioConfiguration()
    config.name = "PedestrianCrossingParking"
    config.type = "PedestrianCrossing"
    config.town = "Town04_Opt"
    # config.other_actors = None

    spawn_locations = []
    for i in range(num_pedestrians):
        loc = carla.Location(
            x=ego_location.x - 2.5,
            y=i * 5 + ego_location.y + 10,
            z=1.0
        )
        spawn_locations.append(loc)

    trigger_location = ego_location
    config.trigger_points = [carla.Transform(trigger_location, carla.Rotation())]

    car_transform = car.actor.get_transform()
    config.ego_vehicles = [ActorConfigurationData(
        car.actor.type_id, 
        car_transform, 
        rolename="hero"
    )]

    scenario = PedestrianCrossingParking(
        world = world,
        ego_vehicles = [car.actor],
        config=config,
        spawn_locations=spawn_locations,
        criteria_enable=True
    )

    return scenario.other_actors, scenario

def car_open_scenario(world, car, ego_location, parked_car):
    config = ScenarioConfiguration()
    config.name = "CarDoorParking"
    config.type = "VehicleOpensDoorTwoWays"
    config.town = "Town04Opt"
    config.other_actors = None

    parked_car_transform = parked_car.get_transform()
    config.other_actors = [ActorConfigurationData(
        parked_car.type_id,
        parked_car_transform,
        rolename="scenario"
    )]

    car_transform = car.actor.get_transform()
    forward = car_transform.get_forward_vector()



    trigger_location = carla.Location(
        x=ego_location.x + forward.x*10,
        y=ego_location.y + forward.y * 10,
        z=ego_location.z
    )
    config.trigger_points = [carla.Transform(trigger_location, carla.Rotation())]

    car_transform = car.actor.get_transform()
    config.ego_vehicles = [ActorConfigurationData(
        car.actor.type_id, 
        car_transform, 
        rolename="hero"
    )]

    

    scenario = VehicleOpensDoorTwoWays(
        world = world,
        ego_vehicles = [car.actor],
        config=config,
        criteria_enable=False
    )

    return scenario.other_actors, scenario

def run_scenario(world, destination_parking_spot, parked_spots, ious, recording_file):
    pedestrians = []
    pedestrian_controllers = []
    scenario_obj = None
    recording_camp = None
    try:
        # load parked cars
        parked_cars, parked_cars_bbs = town04_spawn_parked_cars(world, parked_spots, destination_parking_spot, NUM_RANDOM_CARS)


        
        # load car
        car = town04_spawn_ego_vehicle(world, destination_parking_spot)

        CarlaDataProvider.register_actor(car.actor)
        
        world.tick()
        world.tick()

        recording_cam = car.init_recording(recording_file)

        # HACK: enable perfect perception of parked cars
        car.car.obs = parked_cars_bbs

        # HACK: set lane waypoints to guide parking in adjacent lanes
        car.car.lane_wps = parking_lane_waypoints_Town04
    
        
        # DEBUG: Draw marker
        # world.debug.draw_point(start_location, size=0.5, color=carla.Color(255, 0, 0), life_time=120.0)
        # world.debug.draw_string(start_location, 'PEDESTRIAN', color=carla.Color(255, 0, 0), life_time=120.0)
        # num_pedestrians = random.randint(4,15)
        # spawn_times = [random.randint(0, 500) for _ in range(num_pedestrians)]
        # ped_controller = None

        ego_location = car.actor.get_location()

        ### needed to add this to actually get the car moving
        for i in range(5):
            world.tick()
            car.run_step()


        # pedestrians, scenario = pedestrian_scenario(world, car, ego_location)
        # door_cars, door_scenario = car_open_scenario(world, car, ego_location, parked_cars[random.randint(0,7)])

        destination_loc = parking_vehicle_locations_Town04[destination_parking_spot]
        open_possible = filter(lambda car: world.get_blueprint_library().find(car.type_id).get_attribute('has_dynamic_doors').as_bool(), parked_cars)
        door_car = min(open_possible, key=lambda car: abs(ego_location.x-destination_loc.x) +  abs(ego_location.y-destination_loc.y))
        door_car.set_simulate_physics(True)
        door_behavior_left, door_behavior_right = OpenVehicleDoor(door_car, carla.VehicleDoor.FL), OpenVehicleDoor(door_car, carla.VehicleDoor.FR)
        
        door_opened = False

        pedestrian_controllers = []
        pedestrians, ped_scenario = pedestrian_scenario(world, car, ego_location)

        # for i, ped in enumerate(pedestrians):

        #     loc = ped.get_location()
        #     loc.z = 1.0
        #     ped.set_location(loc)
        #     # controller = PedestrianControl(ped)
        #     # end_location = carla.Location(x=loc.x + 4, y=loc.y, z=loc.z)

        #     # controller.update_target_speed(1.5)
        #     # controller.update_waypoints([carla.Transform(end_location, carla.Rotation())])
        #     ped.set_simulate_physics(True)
        #     # pedestrian_controllers.append(controller)

        # for door_car in door_cars:
        #     loc = door_car.get_location()
        #     loc.x = ego_location.x + 15  # 15m ahead
        #     loc.y = ego_location.y + 3   # 3m to the side
        #     loc.z = 0.5
        #     door_car.set_location(loc)

        # run simulation
        door_open_time = random.randint(0,50)
        i = 0
        while not is_done(car):
            world.tick()

            if i == door_open_time:
                car_loc = car.actor.get_location()
                door_car_loc = door_car.get_location()
                print("DOOR CAR Location", door_car_loc)
                
                # if car_loc.distance(door_car_loc) < 2.0:
                door_behavior_left.initialise()  # opens left door
                door_behavior_right.initialise() # opens right
                # door_behavior.update() # not sure if this is needed
                door_opened = True
            
            if door_opened:
                door_behavior_left.update()
                door_behavior_right.update()


            car.run_step()
            
            car.process_recording_frames()     
            ped_scenario.scenario_tree.tick_once()
            UpdateAllActorControls().update()
            i += 1
            
            # town04_spectator_follow(world, car)

        iou = car.iou()
        ious.append(iou)
        print(f'IOU: {iou}')
    finally:
        recording_cam.destroy()
        if car is not None:
            car.destroy()

        if ped_scenario is not None:
            ped_scenario.remove_all_actors()

        for parked_car in parked_cars:
            if parked_car is not None:
                parked_car.destroy()
        world.tick()
        world.tick()

def main():
    try:
        client = load_client()

        # load map
        world = town04_load(client)

        CarlaDataProvider.set_client(client)
        CarlaDataProvider.set_world(world)

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