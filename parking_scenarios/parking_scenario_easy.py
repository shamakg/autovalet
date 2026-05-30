import importlib
from parking_scenarios.parking_cone import ParkingConeScenario
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ScenarioTriggerer, UpdateAllActorControls
from srunner.scenariomanager.timer import TimeOut
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenarios.background_activity import BackgroundBehavior
from parking_scenarios.vehicle_opens_door_parking import VehicleOpensDoorTwoWaysParking
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenarioconfigs.scenario_configuration import ActorConfigurationData, ScenarioConfiguration
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import py_trees

from carla import Transform
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

class ParkingScenarioEasy(BasicScenario):
    category = "ParkingScenario"
    def __init__(self, world, config, destination, parked, debug_mode=0, criteria_enable=True, mode = None, car_class=None, start_y_offset=0.0, start_x_offset=0.0, spawn_cones=True, spawn_parked=True):


        self.world = world
        self.parked_spots = parked
        
        self.destination_parking_spot = destination


        self.other_actors = []
        
        self.DISTANCE_TRIGGER = 100


        self.config = config
        self.debug_mode = debug_mode
        self.criteria_enable = criteria_enable
        self.spawn_cones = spawn_cones
        self.spawn_parked = spawn_parked
        
        
        # load car
        self.car = town04_spawn_ego_vehicle(world, destination, car_class=car_class, start_y_offset=start_y_offset, start_x_offset=start_x_offset)

        CarlaDataProvider.register_actor(self.car.actor)

        self.world.tick()
        

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

        # HACK: set lane waypoints to guide parking in adjacent lanes
        self.car.car.lane_wps = parking_lane_waypoints_Town04


        if self.spawn_parked:
            self.parked_cars, self.parked_cars_bbs, self.parked_cars_and_spots_bbs = town04_spawn_parked_cars(world, parked, destination, NUM_RANDOM_CARS)
        else:
            # Fully empty lot (e.g. controller GT-following test): no parked cars.
            self.parked_cars, self.parked_cars_bbs, self.parked_cars_and_spots_bbs = [], [], []
        # Tick once so CARLA's synchronous-mode actor registry reflects all newly
        # spawned parked cars before build_scenarios() queries get_actors() and
        # before _get_all_occupied_spots() calls car.get_location(). Without this
        # tick the cone scenario sees an empty vehicles_nearby list and places cones
        # on top of freshly spawned parked vehicles.
        world.tick()

        self.build_scenarios(self.car.actor)

        self.timeout = 140
        super().__init__(
            config.name, [self.car.actor], config, world, debug_mode > 3, False, criteria_enable
        )

        

        self.remove_update()


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
    
    
    
    def _get_all_occupied_spots(self):
        occupied = set(self.parked_spots)
        for car in self.parked_cars:
            car_loc = car.get_location()
            for i, spot_loc in enumerate(parking_vehicle_locations_Town04):
                if abs(car_loc.x - spot_loc.x) < 2.0 and abs(car_loc.y - spot_loc.y) < 2.0:
                    occupied.add(i)
                    break
        return list(occupied)

    def load_parking_cones(self, ego_location):
        """Load parking cones in empty spots"""
        config = ScenarioConfiguration()
        config.name = "ParkingCones"
        config.type = "ParkingCones"
        config.town = "Town04_Opt"

        cones_scenario = ParkingConeScenario(
            world=self.world,
            ego_vehicles=[self.car.actor],
            config=config,
            destination_spot=self.destination_parking_spot,
            parked_spots=self._get_all_occupied_spots(),
            criteria_enable=self.criteria_enable,
        )
        
        self.list_scenarios.append(cones_scenario)
        return cones_scenario


    ### Core Logic Here
    def build_scenarios(self, ego_vehicle, debug=False):
        
        new_scenarios = []
        
        ego_location = self.car.actor.get_location() 
        if ego_location is None:
            return 

        print("EGO LOCATION:", ego_location)

        #-------------------------------------------------------

        ## LOAD PEDESTRIAN

        # new_scenarios.append(self.load_pedestrian(ego_location))

        #-------------------------------------------------------

        ## LOAD DOOR OPEN SCENARIO
        
        # new_scenarios.append(self.load_door_open(ego_location))

         #-------------------------------------------------------

        ## LOAD CUT IN CAR SCENARIO -- Update 2/2: Don't think this is possible
        
        # new_scenarios.append(self.load_cut_in(ego_location))

         #-------------------------------------------------------

        # new_scenarios.append(self.load_opposite_vehicle(ego_location))

         #-------------------------------------------------------

        if self.spawn_cones:
            new_scenarios.append(self.load_parking_cones(ego_location))
        


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

        if not scenario_behaviors:
            scenario_behaviors.append(py_trees.behaviours.Running(name="wait"))

        behavior.add_children(scenario_behaviors)

        return behavior


    def _create_test_criteria(self):
        criteria = [
                CollisionTest(self.ego_vehicles[0], other_actor_type="vehicle", name="VehicleCollisionTest"),
                CollisionTest(self.ego_vehicles[0], other_actor_type="walker", name="PedestrianCollisionTest"),
                ]
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

    def _create_timeout_behavior(self):
        return TimeOut(self.timeout, name="TimeOut")

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


        



        


