from srunner.scenarioconfigs.scenario_configuration import ActorConfigurationData, ScenarioConfiguration
from srunner.scenariomanager.actorcontrols.pedestrian_control import PedestrianControl
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import OpenVehicleDoor, UpdateAllActorControls
from srunner.scenarios.pedestrian_crossing import PedestrianCrossing
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

def pedestrian_scenario(world, car, ego_location):
    ## logic to somehwat try to maintain the Pedestrian abstraction for easier implementation with Carla challenges

    config = ScenarioConfiguration()
    config.name = "PedestrianCrossingParking"
    config.type = "PedestrianCrossing"
    config.town = "Town04_Opt"
    config.other_actors = None

    trigger_location = ego_location ### Doesn't REALLY matter since it's getting overwritten in get_pedestrian_controllers
    config.trigger_points = [carla.Transform(trigger_location, carla.Rotation())]

    config.ego_vehicles = [car.actor]

    scenario = PedestrianCrossing(
        world = world,
        ego_vehicles = [car.actor],
        config=config,
        criteria_enable=True
    )

    return scenario.other_actors, scenario

## THis method, at least two my knowledge, is difficult to implement since it spawns a NEW car (doesnt use existing)
def car_open_scenario(world, car, ego_location, parked_car):
    config = ScenarioConfiguration()
    config.name = "CarDoorParking"
    config.type = "VehicleOpensDoorTwoWays"
    config.town = "Town04_Opt"
    config.other_actors = None

    # parked_car_transform = parked_car.get_transform()
    config.other_actors = [parked_car]

    trigger_location = ego_location ## again, this doesn't really matter too much
    config.trigger_points = [carla.Transform(trigger_location, carla.Rotation())]

    # car_transform = car.actor.get_transform()
    config.ego_vehicles = [car.actor]

    scenario = VehicleOpensDoorTwoWays(
        world = world,
        ego_vehicles = [car.actor],
        config=config,
        criteria_enable=False
    )

    return scenario.other_actors, scenario

def get_pedestrian_controllers(pedestrians, spawn_location, velocity = 1.0, speed = 1.0):
    pedestrian_controllers = []
    # This is the part I'm concerned about because it sort of breaks the Carla abstraction. for example, I am manually changing the Pedestrians
    # location, which means I'm overriding its behavior tree and using a Pedestrian Control (controller) to control it manually,
    for i, ped in enumerate(pedestrians):
            loc = ped.get_location()
            loc.x, loc.y, loc.z = spawn_location.x, i*1 + spawn_location.y, spawn_location.z
            ped.set_location(loc)
            controller = PedestrianControl(ped)
            end_location = carla.Location(x=loc.x + velocity*3, y=loc.y, z=loc.z)
            ped.set_simulate_physics(True)

            controller.update_target_speed(speed)
            controller.update_waypoints([carla.Transform(end_location, carla.Rotation())])
            pedestrian_controllers.append(controller)
    return pedestrian_controllers

def spawn_x_pedestrians(world, car, ego_location, pedestrian_triples=1):
    all_pedestrians = []
    all_scenarios = []
    pedestrian_controllers = []
    for i in range(pedestrian_triples):
        pedestrians, scenario = pedestrian_scenario(world, car, ego_location)
        all_scenarios.append(scenario)
        all_pedestrians.extend(pedestrians)
        ego_y_offset = random.randint(3, 25)
        possibilities = [(1, carla.Location(ego_location.x - 2.2, ego_y_offset + ego_location.y + 10, 1)), (-1, carla.Location(ego_location.x + 2.5, ego_y_offset + ego_location.y + 10, 1))]
        velocity, spawn_location = possibilities[random.randint(0,1)]
        speed = 1
        pedestrian_controllers.extend(get_pedestrian_controllers(pedestrians, spawn_location, velocity = velocity, speed=speed)) ## not actually velocity, but helps find the direction of the pedestrian's motion in the x_direction
    return all_pedestrians, all_scenarios, pedestrian_controllers

def spawn_pedestrian(world, car, ego_location):
    pedestrian_controllers = []
    pedestrians, scenario = pedestrian_scenario(world, car, ego_location)
    ego_y_offset = random.randint(3, 25)
    possibilities = [(1, carla.Location(ego_location.x - 2.2, ego_y_offset + ego_location.y + 10, 1)), (-1, carla.Location(ego_location.x + 2.5, ego_y_offset + ego_location.y + 10, 1))]
    velocity, spawn_location = possibilities[random.randint(0,1)]
    speed = 1
    pedestrian_controllers.extend(get_pedestrian_controllers(pedestrians, spawn_location, velocity = velocity, speed=speed)) ## not actually velocity, but helps find the direction of the pedestrian's motion in the x_direction
    return pedestrians, scenario, pedestrian_controllers

def get_door_behaviors(destination_parking_spot, world, parked_cars, ego_location):
    ## This also breaks the abstraction since it finds the closest open-able car to the destination and manually sets its behavior to DoorOpen
    destination_loc = parking_vehicle_locations_Town04[destination_parking_spot]
    open_possible = list(filter(lambda car: world.get_blueprint_library().find(car.type_id).get_attribute('has_dynamic_doors').as_bool(), parked_cars))
    if len(open_possible) == 0:
        print("No cars are open-able!!!")
        return None, None, None
    door_car = min(open_possible, key=lambda car: abs(ego_location.x-destination_loc.x) +  abs(ego_location.y-destination_loc.y))
    door_car.set_simulate_physics(True)
    door_behavior_left, door_behavior_right = OpenVehicleDoor(door_car, carla.VehicleDoor.FL), OpenVehicleDoor(door_car, carla.VehicleDoor.FR)
    return door_behavior_left, door_behavior_right, door_car


def run_scenario(world, destination_parking_spot, parked_spots, ious, recording_file):
    pedestrians = []
    pedestrian_controllers = []
    scenario_obj = None
    recording_cam = None
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
    
        ego_location = car.actor.get_location()
        

        ### HACK: needed to add this to actually get the car moving 
        for i in range(5):
            # world.tick()
            car.run_step()

        door_scenario = None
        destination_loc = parking_vehicle_locations_Town04[destination_parking_spot]
        open_possible = list(filter(lambda c: world.get_blueprint_library().find(c.type_id).get_attribute('has_dynamic_doors').as_bool(), parked_cars))
        
        CarlaDataProvider.register_actor(car.actor)
        if len(open_possible) > 0:
            door_car = min(open_possible, key=lambda c: abs(ego_location.x-destination_loc.x) + abs(ego_location.y-destination_loc.y))
            CarlaDataProvider.register_actor(door_car)
            door_scenario_actors, door_scenario = car_open_scenario(world, car, ego_location, door_car)
            print(f"✅ Door scenario created")
        
        


       
        # door_cars, door_scenario = car_open_scenario(world, car, ego_location, parked_cars[random.randint(0,7)])


        door_behavior_left, door_behavior_right, door_car = get_door_behaviors(destination_parking_spot, world, parked_cars, ego_location)
        door_opened = False

        ### generate some random pedestrians. the scenario spawns three of them but we can add more groups
        pedestrian_triples = 5 
        
        #### 

        # run simulation
        door_open_time = random.randint(0,75) # get a random door opened time, realistically should be tuned based on what expected run time is
        pedestrian_spawn_times = [random.randint(0, 75) for i in range(random.randint(2, 6))]
        all_pedestrians, all_scenarios, pedestrian_controllers = [], [], []
        # new_pedestrians, new_scenarios, new_pedestrian_controllers = spawn_pedestrian(world, car, ego_location)
        # pedestrians.extend(new_pedestrians)
        # all_scenarios.append(new_scenarios)
        # pedestrian_controllers.extend(new_pedestrian_controllers)

        if door_scenario:
            all_scenarios.append(door_scenario)

        i = 0
        while not is_done(car):
            world.tick()

            # if i == door_open_time:
            #     # car_loc = car.actor.get_location()
            #     door_car_loc = door_car.get_location()
            #     print("DOOR CAR Location", door_car_loc)
                
            #     # if car_loc.distance(door_car_loc) < 2.0:
            #     door_behavior_left.initialise()  # opens left door
            #     door_behavior_right.initialise() # opens right
            #     # door_behavior.update() # not sure if this is needed
            #     door_opened = True

            if i in pedestrian_spawn_times:
                new_pedestrians, new_scenarios, new_pedestrian_controllers = spawn_pedestrian(world, car, ego_location)
                pedestrians.extend(new_pedestrians)
                all_scenarios.append(new_scenarios)
                pedestrian_controllers.extend(new_pedestrian_controllers)
            
            # if door_opened:
            #     door_behavior_left.update()
            #     door_behavior_right.update()


            car.run_step()
            
            if door_scenario:
                door_scenario.scenario_tree.tick_once()
                
                # for criterion in scenario.get_criteria():
                #     criterion_id = id(criterion)
                #     if hasattr(criterion, 'test_status') and criterion.test_status == "FAILURE":
                #         if criterion_id not in collision_logged:
                #             collision_logged.add(criterion_id)
                #             print(f"\n🚨 COLLISION! Scenario: {scenario.name}\n")
            
            # car.process_recording_frames()   

            car.process_recording_frames()     
            for controller in pedestrian_controllers:
                controller.run_step()
            i += 1
            
            # town04_spectator_follow(world, car)

        iou = car.iou()
        ious.append(iou)
        print(f'IOU: {iou}')
    finally:
        recording_cam.destroy()
        if car is not None:
            car.destroy()

        # for pedestrian in pedestrians:
        #     if pedestrian is not None and pedestrian.is_alive:
        #         pedestrian.destroy()

        for parked_car in parked_cars:
            if parked_car is not None:
                parked_car.destroy()

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