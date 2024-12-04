### Version 3 of the code, with the goal being to renew the pedestrian crossing abstraction

from leaderboard.autovalet.parking_scenarios.parking_scenario import ParkingScenario
from srunner.scenarioconfigs.scenario_configuration import ActorConfigurationData, ScenarioConfiguration
from srunner.scenariomanager.actorcontrols.pedestrian_control import PedestrianControl
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.result_writer import ResultOutputProvider
from srunner.scenariomanager.scenario_manager import ScenarioManager
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import OpenVehicleDoor, UpdateAllActorControls
import py_trees
from v2_experiment_utils import (
    load_client,
    is_done,
    town04_load,
    town04_spectator_bev,

)
import time

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

### CLAUDED positions
def get_pedestrian_locations(destination_parking_spot):
    """
    Create realistic, crowded pedestrian positions crossing laterally across the parking lane.
    Ensures pedestrians don't spawn inside vehicles or too close to each other.
    """
    spawn_locations = []
    destination_loc = parking_vehicle_locations_Town04[destination_parking_spot]
    
    # Get the world to check for collisions
    world = CarlaDataProvider.get_world()
    
    def is_valid_spawn_location(loc, min_distance=1.5):
        # Check distance to all existing pedestrian spawn points
        for existing_loc, _, _ in spawn_locations:
            dist = np.sqrt((loc.x - existing_loc.x)**2 + (loc.y - existing_loc.y)**2)
            if dist < min_distance:
                return False
        
        return True
    
    def try_spawn_pedestrian(base_x, y_min, y_max, direction_base, direction_variance, max_attempts=50, velocity = CarlaDataProvider.get_random_seed()
.uniform(1.3, 2.0)):

        for _ in range(max_attempts):
            loc = carla.Location(
                x=base_x,  # Small x variation
                y=destination_loc.y + random.uniform(y_min, y_max),
                z=1.0
            )
            
            if is_valid_spawn_location(loc):
                direction = direction_base + random.uniform(-direction_variance, direction_variance)
                return (loc, direction, velocity)
        
        return None
    
    # Group 1: Close-range crossers (immediately in front of destination)
    for i in range(4):
        result = try_spawn_pedestrian(
            base_x=destination_loc.x - 3.25,
            y_min=-2, y_max=8,
            direction_base=270, direction_variance=15
        )
        if result:
            spawn_locations.append(result)
    
    # Group 2: Mid-range crossers (spread across the approach path)
    for i in range(5):
        result = try_spawn_pedestrian(
            base_x=destination_loc.x - 3.25,
            y_min=-5, y_max=12,
            direction_base=270, direction_variance=20
        )
        if result:
            spawn_locations.append(result)
    
    # Group 3: Far-range crossers (creating ongoing traffic)
    for i in range(3):
        result = try_spawn_pedestrian(
            base_x=destination_loc.x - 3.25,
            y_min=10, y_max=25,
            direction_base=270, direction_variance=10
        )
        if result:
            spawn_locations.append(result)
    
    # Group 4: Opposite direction crossers
    for i in range(3):
        result = try_spawn_pedestrian(
            base_x=destination_loc.x - 7.25,
            y_min=-3, y_max=10,
            direction_base=90, direction_variance=15
        )
        if result:
            spawn_locations.append(result)

    for i in range(3):
        result = try_spawn_pedestrian(
            base_x=destination_loc.x - 7.25,
            y_min=-3, y_max=10,
            direction_base=0, direction_variance=15
        )
    if result:
        spawn_locations.append(result)

    for i in range(3):
        result = try_spawn_pedestrian(
            base_x=destination_loc.x - 3.25,
            y_min=-3, y_max=10,
            direction_base=180, direction_variance=15
        )
        if result:
            spawn_locations.append(result)

    # DEBUG: Print all spawn points
    for i, (loc, direction, velocity) in enumerate(spawn_locations):
        print(f"Pedestrian {i}: pos=({loc.x}, {loc.y}), direction={direction:.1f}°")
    
    return spawn_locations

class ParkingAgent:
    """Simple agent that controls the parking car"""
    
    def __init__(self, parking_car, recording_file):
        self.parking_car = parking_car
        self.recording_file = recording_file

    def sensors(self):
        """Return list of sensors - required by AgentWrapper"""
        return []  # No sensors needed, car handles perception internally
    
    
    def setup_sensors(self, ego_vehicle, debug_mode):
        """Called by ScenarioManager"""
        pass
    
    def __call__(self):
        """Called each tick to get vehicle control"""
        # Run the parking car's step
        control = self.parking_car.run_step()
        
        # Process recording frames
        self.parking_car.process_recording_frames()
        
        # Return empty control since car handles it internally
        return control
    
    def cleanup(self):
        """Called on scenario end"""
        pass


def run_scenario(world, destination_parking_spot, parked_spots, ious, recording_file):
    recording_cam = None
    parking_scenario = None
    scenario_manager = None
    try:  
        pedestrian_locations= get_pedestrian_locations(destination_parking_spot)

        config = ScenarioConfiguration()
        config.name = "Parking"
        config.type = "Parking"
        config.town = "Town04_Opt"
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

        agent = ParkingAgent(parking_scenario.car, recording_file)

        scenario_manager = ScenarioManager(
            debug_mode=False,
            sync_mode=True,  # Synchronous mode
            timeout=60.0
        )

        
        world.tick()
        world.tick()
        world.tick()

        scenario_manager.load_scenario(
            scenario=parking_scenario,
            agent=agent
        )

        
        # Run the scenario - this handles all ticking internally
        scenario_manager.run_scenario()

        while not is_done(parking_scenario.car):
            time.sleep(0.1)
        
        # Analyze results
        scenario_manager.analyze_scenario(
            stdout=True,
            filename="",
            junit=False,
            json=False
        )

        
        
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