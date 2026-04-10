import importlib
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, ScenarioTimeoutTest
from srunner.scenarios.background_activity import BackgroundBehavior
from srunner.scenarios.opposite_vehicle_parking import OppositeDirectionVehicle
from srunner.scenarios.parking_cut_in_parking import ParkingCutInParking
from parking_scenarios.vehicle_opens_door_parking import VehicleOpensDoorTwoWaysParking
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenarioconfigs.scenario_configuration import ActorConfigurationData, ScenarioConfiguration
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import py_trees

from carla import Transform
from parking_scenarios.pedestrian_crossing_parking import PedestrianCrossingParking
from testbed.v2_experiment_utils import (
    town04_spawn_ego_vehicle,
    town04_spawn_parked_cars,
)

# For lane waypoint hack
from parking_position import (
    parking_lane_waypoints_Town04,
    parking_vehicle_locations_Town04,
)



import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import carla
import glob
import os
import inspect
import sys

### Some Sample Challenge Scenarios
from srunner.scenarios.vehicle_opens_door import VehicleOpensDoorTwoWays
from srunner.scenarios.pedestrian_crossing import PedestrianCrossing
from srunner.scenarios.cross_bicycle_flow import CrossingBicycleFlow
from srunner.scenarios.parking_cut_in import ParkingCutIn
from srunner.scenarios.parking_exit import ParkingExit

### Sample Triggers
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import InTriggerRegion
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import InTriggerDistanceToVehicle
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import InTriggerDistanceToLocation

from v2_experiment import NUM_RANDOM_CARS

# The idea of this class is to be a combination of RouteScenario and V2_Experiment
# The RouteScenario Class has the following instance variables: the world, a config (a ScenarioConfiguration)
# containing info like name, type, town, route, ego_vehicles, and most importantly, scenario_configs which are 
# the smaller challenges within a specific route. In ParkingScenario, there is no "route" but these challenges 
# are crucial
# #### Route Scenario Function Parallels
# spawn_ego_vehicle: IMPLEMENTED - pretty much the same as RouteScenario, but I've used the town04_spawn_ego_vehicle method instead
# ## Unsure about world.tick() after spawning car? This is there in RouteScenario but not v2_experiment
# _get_route: It's possible to get the Hybrid A * route, but not sure if it's needed when the car has its own run_step function
# filter_scenario: Could be used, but unnecessary (will hard code challenge scenarios)
# get_parking_slots: Not needed for now, we are looping through the scenario list in the Runner (parking_scenario_runner)
# spawn_parked_vehicles: IMPLEMENTED - self.close/self.is free not necessary, since we know which ones are occupied/free
# ## at least for this scenario. Added v2_experiment logic
# Now the KEY PROBLEM to add these challenges to a parking scenario is assimilating the following functions
# ## get_all_scenario_classes, build_scenarios, initialize_actors, create_behavior, create_test_criteria
### Create_behavior: IMPLEMENTED

class ParkingScenario(BasicScenario):
    category = "ParkingScenario"
    def __init__(self, world, config, destination, parked, debug_mode=0, criteria_enable=True):

        self.config = config
        self.world = world
        self.parked_spots = parked    
        self.destination_parking_spot = destination
        self.ped_trigger_dist = 100

        self.other_actors = []
        
        self.DISTANCE_TRIGGER = 100


        self.parked_cars, self.parked_cars_bbs = town04_spawn_parked_cars(world, parked, destination, NUM_RANDOM_CARS)

        self.criteria_enable = criteria_enable
        
        # load car
        self.car = town04_spawn_ego_vehicle(world, destination)

        CarlaDataProvider.register_actor(self.car.actor)

        # HACK: enable perfect perception of parked cars
        self.car.car.obs = self.parked_cars_bbs

        # HACK: set lane waypoints to guide parking in adjacent lanes
        self.car.car.lane_wps = parking_lane_waypoints_Town04
        ### Recording Logic
        # self.record()
        world.tick()
        

        self.car.run_step()
        
        self.timeout = 60

        # self.all_scenario_classes = None
        # self.ego_data = None

        # self.scenario_triggerer = None
        # # self.behavior_node = None # behavior node created by _create_behavior()
        # # self.criteria_node = None # criteria node created by _create_test_criteria()

        self.list_scenarios = []
        # self.occupied_parking_locations = []
        # self.available_parking_locations = []

        # scenario_configurations = self._filter_scenarios(config.scenario_configs) ## can add logic to modularize the configs later
        # self.scenario_configurations = scenario_configurations
        # self.missing_scenario_configurations = scenario_configurations.copy()

        if self.car is None:
            raise ValueError("Shutting down, couldn't spawn the ego vehicle")

        # self.world.tick()
        self.build_scenarios(self.car.actor)

        for _ in range(10):
            self.world.tick()
            

        super().__init__(
            config.name, [self.car.actor], config, world, debug_mode > 3, False, criteria_enable
        )

        # self.remove_update()
        self.world.tick()


    def remove_update(self):
        pass
        # keep = []
        # for child in self.scenario_tree.children:
        #     if not isinstance(child, UpdateAllActorControls):
        #         keep.append(child)
        # self.scenario_tree.children = keep
    
    def _spawn_ego_vehicle(self):
        """Spawn the ego vehicle at the first waypoint of the route"""

        car = town04_spawn_ego_vehicle(self.world, self.destination_parking_spot)
        if not car:
            return
        
        self.world.tick() ### Todo: double check

        return car
    
    def spawn_parked_vehicles(self, ego_vehicle):

        ### Logic from v2_experiment run_scenario
        self.parked_cars, self.parked_cars_bbs = town04_spawn_parked_cars(self.world, self.parked_spots, self.destination_parking_spot, NUM_RANDOM_CARS)
    
    def load_pedestrian (self, ego_location):
        config = ScenarioConfiguration()
        config.name = "PedestrianCrossingParking"
        config.type = "PedestrianCrossing"
        config.town = "Town04_Opt"

        min_trigger_dist = self.DISTANCE_TRIGGER
        distance_to_walk = 15
        pedestrian_scenario = PedestrianCrossingParking(self.world, [self.car.actor], config, self.spawn_locations, 
            parking_vehicle_locations_Town04[self.destination_parking_spot], min_trigger_dist=min_trigger_dist, 
            walker_distance = distance_to_walk, criteria_enable=self.criteria_enable)
        
        self.list_scenarios.append(pedestrian_scenario)
        return pedestrian_scenario
    
    def load_opposite_vehicle(self, ego_location):
        """Load opposite direction vehicle scenario"""
        config = ScenarioConfiguration()
        config.name = "OppositeVehicle"
        config.type = "OppositeVehicle"
        config.town = "Town04_Opt"
        
        destination_loc = parking_vehicle_locations_Town04[self.destination_parking_spot]
        
        # Spawn vehicle 40 meters ahead in opposite direction
        spawn_location = carla.Location(
            x=ego_location.x + 1.6,  # Ahead of destination
            y=destination_loc.y + 17,   # Opposite lane (offset to left)
            z=1
        )
        
        # Drive towards ego (past the destination)
        target_location = carla.Location(
            x=ego_location.x,  # Drive past ego
            y=destination_loc.y - 5,
            z=1
        )
        
        opposite_scenario = OppositeDirectionVehicle(
            world=self.world,
            ego_vehicles=[self.car.actor],
            config=config,
            spawn_location=spawn_location,
            target_location=target_location,
            speed=5.0,  # 5 m/s = 18 km/h
            trigger_distance=50.0,  # Start when ego within 30m
            criteria_enable=self.criteria_enable
        )
        
        self.list_scenarios.append(opposite_scenario)
        
        # Debug visualization
        # self.world.debug.draw_point(
        #     spawn_location,
        #     size=0.5,
        #     color=carla.Color(0, 255, 255),
        #     life_time=self.timeout
        # )
        
        return opposite_scenario

    def load_door_open(self, ego_location):
        destination_loc = parking_vehicle_locations_Town04[self.destination_parking_spot]
        open_possible = list(filter(
            lambda c: self.world.get_blueprint_library().find(c.type_id).get_attribute('has_dynamic_doors').as_bool(), 
            self.parked_cars
        ))

        if len(open_possible) == 0:
            return
        
        door_car = min(open_possible, key=lambda c: 
            abs(ego_location.x - destination_loc.x) + abs(ego_location.y - destination_loc.y)
        )
        
        door_config = ScenarioConfiguration()
        door_config.name = "CarDoorParking"
        door_config.type = "VehicleOpensDoorTwoWaysParking"
        door_config.town = "Town04_Opt"
        door_config.trigger_points = [Transform(ego_location)]
        door_config.other_actors = [door_car]
            
        door_scenario = VehicleOpensDoorTwoWaysParking(
            self.world,
            [self.car.actor],
            door_car,
            door_config,
            criteria_enable=self.criteria_enable 
        )
        
        self.list_scenarios.append(door_scenario)
        
        # Debug: draw the door car location
        # self.world.debug.draw_point(
        #     door_car.get_location(), size=0.5, color=carla.Color(255, 0, 0), life_time=10
        # )
    
        return door_scenario
    
    def load_cut_in(self, ego_location):
        destination_loc = parking_vehicle_locations_Town04[self.destination_parking_spot]
        cut_in_car = list(sorted(self.parked_cars, key=lambda c: 
            abs(ego_location.x - destination_loc.x) + abs(ego_location.y - destination_loc.y)
        ))[1]

        door_config = ScenarioConfiguration()
        door_config.name = "CutInParking"
        door_config.type = "CutInParking"
        door_config.town = "Town04_Opt"
        door_config.trigger_points = [Transform(ego_location)]
        door_config.other_actors = [cut_in_car]
             
        cut_in_scenario = ParkingCutInParking(
            self.world,
            [self.car.actor],
            cut_in_car,
            door_config,
            criteria_enable=self.criteria_enable 
        )
        
        self.list_scenarios.append(cut_in_scenario)

        # Debug: draw the cut in car location
        # self.world.debug.draw_point(
        #     cut_in_car.get_location(), size=0.5, color=carla.Color(255, 0, 0), life_time=10
        # )
    
        return cut_in_scenario

    
    ### Core Logic Here
    def build_scenarios(self, ego_vehicle, debug=False):
        
        new_scenarios = []
        
        ego_location = self.car.actor.get_location() 
        if ego_location is None:
            return 

        print("EGO LOCATION:", ego_location)

        #-------------------------------------------------------

        ## LOAD PEDESTRIAN

        new_scenarios.append(self.load_pedestrian(ego_location))

        #-------------------------------------------------------

        ## LOAD DOOR OPEN SCENARIO
        
        new_scenarios.append(self.load_door_open(ego_location))

         #-------------------------------------------------------

        ## LOAD CUT IN CAR SCENARIO -- Update 2/2: Don't think this is possible
        
        # new_scenarios.append(self.load_cut_in(ego_location))

         #-------------------------------------------------------

        new_scenarios.append(self.load_opposite_vehicle(ego_location))
        


    def _initialize_actors(self, config):
        """
        Set other_actors to the superset of all scenario actors
        """
        # Add all the actors of the specific scenarios to self.other_actors
        for scenario in self.list_scenarios:
            self.other_actors.extend(scenario.other_actors)

    def _create_behavior(self):

        behavior = py_trees.composites.Parallel(name="Parking Behavior", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)

        self.behavior_node = behavior
        scenario_behaviors = []

        ### THIS is important: will get all sub scenarios
        for scenario in self.list_scenarios: 
            if scenario.behavior_tree is not None:    
                scenario_behaviors.append(scenario.behavior_tree)


        ### I dont think we need a scenario triggerer since this is running immediately
        self.scenario_triggerer = None

        # # Add the Background Activity -- if we want can add extra cars
        # behavior.add_child(BackgroundBehavior(self.car.actor, dummy_route, name="BackgroundActivity"))

        behavior.add_children(scenario_behaviors)

        return behavior


    def _create_test_criteria(self):
        criteria = [ScenarioTimeoutTest(self.ego_vehicles[0], self.config.name), CollisionTest(self.ego_vehicles[0])]
        for scenario in self.list_scenarios:
            if hasattr(scenario, "_create_test_criteria"):
                sub_criteria = scenario.get_criteria()
                if sub_criteria:
                    criteria.extend(sub_criteria)

        return criteria

    
    def _setup_scenario_trigger(self, config):
        return None
    
    def _setup_scenario_end(self, config):
        return None
    
    ## From Route Scenario
    def cleanup(self):
        """
        Remove all actors upon deletion
        """

        ### From v2_experiment
        if self.car.actor and self.car.actor.is_alive:
            try:
                self.car.destroy()
            except RuntimeError:
                pass

        for scenario in self.list_scenarios:
            if hasattr(scenario, 'other_actors'):
                for actor in scenario.other_actors:
                    if actor and actor.is_alive:
                        try:
                            actor.destroy()
                        except:
                            pass


        for parked_car in self.parked_cars:
            if parked_car and parked_car.is_alive:
                try:
                    parked_car.destroy()
                except:
                    pass


        



        


