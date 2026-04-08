from enum import Enum
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import UpdateAllActorControls
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, ScenarioTimeoutTest
from parking_scenarios.vehicle_opens_door_parking import VehicleOpensDoorTwoWaysParking
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenarioconfigs.scenario_configuration import ActorConfigurationData, ScenarioConfiguration
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import py_trees

from carla import Transform
from parking_scenarios.pedestrian_crossing_parking import PedestrianCrossingParking
from parking_scenarios.smart_pedestrian_crossing_parking import SmartPedestrianCrossingParking

from testbed.v2_experiment_utils import (
    town04_spawn_ego_vehicle,
    town04_spawn_parked_cars,
    town04_spawn_parked_cars_with_doors,
)

# For lane waypoint hack
from parking_position import (
    parking_lane_waypoints_Town04,
    parking_vehicle_locations_Town04,
)

import carla
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

class HardMode(Enum):
    DoorMode = 0
    PedMode = 1

class ParkingScenarioHard(BasicScenario):
    category = "ParkingScenarioHard"
    def __init__(self, world, config, destination, parked, debug_mode=0, criteria_enable=True, mode = HardMode.PedMode, car_class=None):

        self.config = config
        self.world = world
        self.parked_spots = parked
        # self.ped_spawner = get_pedestrian_locations_medium_1
        # self.spawn_locations = self.ped_spawner(destination)      
        self.destination_parking_spot = destination
        self.ped_trigger_dist = 100

        self.mode = mode
        self.other_actors = []
        
        self.DISTANCE_TRIGGER = 100

        
        self.config = config
        self.debug_mode = debug_mode
        self.criteria_enable = criteria_enable
        
        
        # load car
        self.car = town04_spawn_ego_vehicle(world, destination, car_class=car_class)

        CarlaDataProvider.register_actor(self.car.actor)

        self.world.tick()
        
        self.timeout = 70

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

        if mode == HardMode.DoorMode:
            self.parked_cars, self.parked_cars_bbs, self.parked_cars_and_spots_bbs = town04_spawn_parked_cars_with_doors(world, parked, destination, NUM_RANDOM_CARS)
        else:
            self.parked_cars, self.parked_cars_bbs, self.parked_cars_and_spots_bbs = town04_spawn_parked_cars(world, parked, destination, NUM_RANDOM_CARS)
        
        self.build_scenarios(self.car.actor)

        super().__init__(
            config.name, [self.car.actor], config, world, debug_mode > 3, False, criteria_enable
        )

        

        self.remove_update()
        


    def remove_update(self):
        # pass
        keep = []
        for child in self.scenario_tree.children:
            if not isinstance(child, UpdateAllActorControls):
                keep.append(child)
        self.scenario_tree.children = keep
    
    def _spawn_ego_vehicle(self):
        """Spawn the ego vehicle at the first waypoint of the route"""

        car = town04_spawn_ego_vehicle(self.world, self.destination_parking_spot)
        if not car:
            return
        
        self.world.tick() ### Todo: double check

        return car
    
    # def spawn_parked_vehicles(self, ego_vehicle):

    #     ### Logic from v2_experiment run_scenario
    #     self.parked_cars, self.parked_cars_bbs = town04_spawn_parked_cars(self.world, self.parked_spots, self.destination_parking_spot, NUM_RANDOM_CARS)
    
    def load_pedestrian (self, ego_location):
        config = ScenarioConfiguration()
        config.name = "PedestrianCrossingParking"
        config.type = "PedestrianCrossing"
        config.town = "Town04_Opt"

        min_trigger_dist = 16.5
        distance_to_walk = 15
        pedestrian_scenario = PedestrianCrossingParking(self.world, [self.car.actor], config, self.destination_parking_spot, min_trigger_dist=min_trigger_dist, 
            criteria_enable=self.criteria_enable)
        
        self.list_scenarios.append(pedestrian_scenario)
        return pedestrian_scenario
    

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
        self.world.debug.draw_point(
            door_car.get_location(), size=0.5, color=carla.Color(255, 0, 0), life_time=10
        )
    
        return door_scenario
    

    
    ### Core Logic Here
    def build_scenarios(self, ego_vehicle, debug=False):
        
        new_scenarios = []
        
        ego_location = self.car.actor.get_location() 
        if ego_location is None:
            return 

        print("EGO LOCATION:", ego_location)

        #-------------------------------------------------------

        ## LOAD PEDESTRIAN

        if self.mode == HardMode.PedMode:
            new_scenarios.append(self.load_pedestrian(ego_location))
        
        elif self.mode == HardMode.DoorMode:
            new_scenarios.append(self.load_door_open(ego_location))

        #-------------------------------------------------------

        # super().__init__(
        #     self.config.name, [self.car.actor], self.config, self.world, self.debug_mode > 3, False, self.criteria_enable
        # )

        # self.debug_tree(self.scenario_tree)



    def _initialize_actors(self, config):
        print(f"DEBUG: Before extending other_actors")
        for scenario in self.list_scenarios:
            print(f"  Scenario {scenario.name} has {len(scenario.other_actors)} other_actors")
            for actor in scenario.other_actors:
                print(f"    Actor ID: {actor.id}, Type: {actor.type_id}")
                if actor.id == self.ego_vehicles[0].id:
                    print(f"    ⚠️  WARNING: EGO VEHICLE FOUND IN OTHER_ACTORS!")
            self.other_actors.extend(scenario.other_actors)

    def debug_tree(self, node, depth=0):
        print("  " * depth + f"- {type(node)}")
        if hasattr(node, "children"):
            for c in node.children:
                self.debug_tree(c, depth+1)

    def _initialize_environment(self, world):
        # Skip the parent's initialization that ticks the world
        pass

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
        criteria = [
                ScenarioTimeoutTest(self.ego_vehicles[0], self.config.name),
                CollisionTest(self.ego_vehicles[0], other_actor_type="vehicle", name="VehicleCollisionTest"),
                CollisionTest(self.ego_vehicles[0], other_actor_type="walker", name="PedestrianCollisionTest"),
                CollisionTest(self.ego_vehicles[0], other_actor_type="prop", name="PropCollisionTest"),
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


        



        


