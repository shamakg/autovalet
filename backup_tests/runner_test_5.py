### Version 3 of the code, with the goal being to renew the pedestrian crossing abstraction

from leaderboard.autovalet.parking_scenarios.parking_scenario import ParkingScenario
from srunner.scenarioconfigs.scenario_configuration import ActorConfigurationData, ScenarioConfiguration
from srunner.scenariomanager.actorcontrols.pedestrian_control import PedestrianControl
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.result_writer import ResultOutputProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import OpenVehicleDoor, UpdateAllActorControls
import py_trees
from srunner.scenarios.pedestrian_crossing import PedestrianCrossing
from v2_experiment_utils import (
    load_client,
    is_done,
    town04_load,
    town04_spawn_ego_vehicle,
    town04_spawn_parked_cars,
    town04_spectator_bev,

)

from parking_position import (
    parking_vehicle_locations_Town04,
)

# For lane waypoint hack
from parking_position import (
    parking_lane_waypoints_Town04
)

import carla
import time

import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio

from srunner.scenariomanager.actorcontrols.basic_control import BasicControl
import random

from v2_experiment import SCENARIOS
NUM_RANDOM_CARS = 25


def get_pedestrian_locations(destination_parking_spot):
    ### We can add pedestrians to any spawn point in the parking lot (they will start moving at random velocities in the specified direction)
    num_pedestrians = 5
    spawn_locations = []
    destination_loc = parking_vehicle_locations_Town04[destination_parking_spot]
    for i in range(num_pedestrians):
        loc = carla.Location(
            x=destination_loc.x - 3.25,
            y=destination_loc.y + 3*i + 5.0,
            z=1.0
        )
        ped_direction = 270 - i * 10
        spawn_locations.append((loc, ped_direction))
        
        # DEBUG: Draw spawn points
        print(f"Pedestrian {i} spawn location: {loc}")

    return spawn_locations

def run_scenario(world, destination_parking_spot, parked_spots, ious, recording_file):
    recording_cam = None
    parking_scenario = None
    try:  
        pedestrian_locations= get_pedestrian_locations(destination_parking_spot)

        config = ScenarioConfiguration()
        config.name = "Parking"
        config.type = "Parking"
        config.town = "Town04"
        config.other_actors = None
        parking_scenario = ParkingScenario(
            world=world,
            config=config,
            # car=car,
            spawn_locations = pedestrian_locations,
            destination=destination_parking_spot,
            parked=parked_spots,
            debug_mode=0,
            criteria_enable=True,
        )


        recording_cam = parking_scenario.car.init_recording(recording_file)

        world.tick()

        
        while not is_done(parking_scenario.car):
            
            world.tick()
            CarlaDataProvider.on_carla_tick() ### NEED THIS so that trigger distance will work

            parking_scenario.car.run_step()

            if parking_scenario.scenario_tree:
                parking_scenario.scenario_tree.tick_once()

            parking_scenario.car.process_recording_frames()    
            
            # town04_spectator_follow(world, car)
        
        iou = parking_scenario.car.iou()
        ious.append(iou)
        print(f'IOU: {iou}')
    finally:
        analyze_scenario(parking_scenario)
        if recording_cam:
            recording_cam.destroy()
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
        
        for criterion in parking_scenario.get_criteria():
            criterion.terminate(None)
            print(criterion.name, criterion.test_status)
            if criterion.test_status != "SUCCESS":
                result = "SCENARIO FAILURE"

        print("FINAL SCENARIO RESULT:", result)
        return result
        # parking_scenario.scenario_tree.terminate(py_trees.common.Status.SUCCESS)

def main():
    try:
        client = load_client()

        # load map
        world = client.load_world('LargeParkingLotv1')

        carla_map = world.get_map()

        default = carla_map.get_spawn_points()[0]

        print(default.location.x, default.location.y, default.location.z)

        spectator_location = carla.Location(default.location.x, default.location.y, z=40)
        spectator_rotation = carla.Rotation(pitch=-90.0)
        world.get_spectator().set_transform(carla.Transform(spectator_location, spectator_rotation))

        
        waypoints = carla_map.generate_waypoints(1.0)
        print(f"Number of waypoints: {len(waypoints)}")

        xs = [wp.transform.location.x for wp in waypoints]
        ys = [wp.transform.location.y for wp in waypoints]


        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,10))
        plt.scatter(xs, ys, s=1, c='blue')
        plt.title("Waypoint map of LargeParkingLotv1")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis('equal')
        plt.savefig('waypoints.png')


        config = ScenarioConfiguration()
        config.name = "PedestrianCrossingParking"
        config.type = "PedestrianCrossing"
        config.town = "Town04_Opt"

        min_trigger_dist = 15.0
        distance_to_walk = 15
        # pedestrian_scenario = PedestrianCrossing(self.world, [self.car.actor], config, criteria_enable=self.criteria_enable)
        parked_cars = town04_spawn_parked_cars(world, SCENARIOS[0][1], SCENARIOS[0][0], NUM_RANDOM_CARS)
        car = town04_spawn_ego_vehicle(world, SCENARIOS[0][0])
        # load spectator
        # town04_spectator_bev(world)

        # # load recording file
        # recording_file = iio.imopen('./test.mp4', 'w', plugin='pyav')
        # recording_file.init_video_stream('vp9', fps=30)

        # # run scenarios
        ious = []
        while True:
            time.sleep(0.1)
        
        # for destination_parking_spot, parked_spots in SCENARIOS:
        #     print(f'Running scenario: destination={destination_parking_spot}, parked_spots={parked_spots}')
        #     # run_scenario(world, destination_parking_spot, parked_spots, ious, recording_file)
        #     pedestrian_locations= get_pedestrian_locations(destination_parking_spot)

        #     config = ScenarioConfiguration()
        #     config.name = "Parking"
        #     config.type = "Parking"
        #     config.town = "Town04_Opt"
        #     config.other_actors = None
        #     parking_scenario = ParkingScenario(
        #         world=world,
        #         config=config,
        #         # car=car,
        #         spawn_locations = pedestrian_locations,
        #         destination=destination_parking_spot,
        #         parked=parked_spots,
        #         debug_mode=0,
        #         criteria_enable=True,
        #     )
        #     break

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
        # recording_file.close()
        world.tick()

if __name__ == '__main__':
    main()