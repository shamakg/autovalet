#!/usr/bin/env python
"""
Scenario where a vehicle drives in the opposite direction
"""

from enum import Enum
import py_trees
import carla

from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorTransformSetter,
    KeepVelocity,
    StopVehicle,
    WaitForever
)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    InTriggerDistanceToLocation
)


from parking_position import (
    parking_lane_waypoints_Town04,
    parking_vehicle_locations_Town04,
    town04_bound
)

from v2 import MIN_SPEED, MAX_SPEED

class CollisionMode(Enum):
    COLLIDE = 0 
    STOP_EARLY = 1
    MISS = 2

class OppositeDirectionVehicle(BasicScenario):

    def __init__(self, world, ego_vehicles, config, destination,
                 collision_mode = CollisionMode.COLLIDE, speed=3.0, trigger_distance=40.0, 
                 debug_mode=False, criteria_enable=False, timeout=60):

        self.world = world
        self.speed = speed

        self.speed = (MIN_SPEED + MAX_SPEED) * 0.75
        self.trigger_distance = trigger_distance
        self.timeout = timeout
        self.opposite_vehicle = None

        self.collision_mode = collision_mode
        self.destination_loc = parking_vehicle_locations_Town04[destination]

        #----------------------------------------------

        # Hardcoded Scenario Constants
        self.yaw = 270 # this is hardcoded, assuming the parked car is moving from left to right
        self.brake_speed = 1.5 # hardcoded, how effective the brakes are
        self.lane_width = 3
        self.start_offset = 25
        self.drive_distance = 100 ## just enough to clear the map, this doesn't affect the scenario
        self.miss_offset = 6 ## space of two cars, miss scenario needs a larger offset

        #----------------------------------------------

        self.spawn_transform = self.get_vehicle_location(ego_vehicles)     
        
        super().__init__(
            name="OppositeDirectionVehicle",
            ego_vehicles=ego_vehicles,
            config=config,
            world=world,
            debug_mode=debug_mode,
            criteria_enable=criteria_enable
        )

    def _initialize_actors(self, config):
        vehicle = CarlaDataProvider.request_new_actor('vehicle.tesla.model3', self.spawn_transform, actor_category="car")
        vehicle.set_location(self.spawn_transform.location + carla.Location(z=-200))
        self.opposite_vehicle = vehicle
        self.other_actors.append(vehicle)

        import math
        yaw_rad = math.radians(self.yaw)
        cross_dir = [math.cos(yaw_rad), math.sin(yaw_rad)]
        loc = self.spawn_transform.location
        ext = vehicle.bounding_box.extent
        # p_cross=1/3: prior — COLLIDE, STOP_EARLY, NEAR_MISS are equiprobable;
        # only COLLIDE is a true crossing event.
        # cross_distance = spawn-to-crossing-point (not drive_distance which is 100m
        # and would spread the 24 time samples far off the ±18m grid).
        # cone_rate=0.05: vehicles stay on-road, lateral spread is much smaller than pedestrians.
        self.gmm_candidate_futures = [{
            'edge_point':     [loc.x, loc.y],
            'cross_dir':      cross_dir,
            'cross_distance': self.cross_distance_m,
            'speed':          float(self.speed),
            'p_cross':        1.0 / 3.0,
            'extent':         [float(ext.x), float(ext.y)],
            'cone_rate':      0.05,
        }]
        # Note: 'longitudinal' flag no longer needed — travel_dir is embedded in
        # each GMM mode by build_crossing_snapshots, which handles box orientation.

    def _create_behavior(self):
        """
        Vehicle waits for ego to get close, then drives to target
        """
        sequence = py_trees.composites.Sequence(name="OppositeVehicleBehavior")
        sequence.add_child(ActorTransformSetter(self.other_actors[0], self.spawn_transform, True))
        # wait for our car to get close
        car_trigger = py_trees.composites.Parallel(
                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        car_trigger.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], 
            self.spawn_transform.location, 
            self.trigger_distance))
        sequence.add_child(car_trigger)
        # drive at constant speed to target
        sequence.add_child(KeepVelocity(
            self.other_actors[0],
            self.speed,
            True,
            duration=self.drive_distance / self.speed,
            distance=self.drive_distance
        ))

        sequence.add_child(StopVehicle(self.other_actors[0], self.brake_speed))
        

        sequence.add_child(WaitForever()) # to ensure the scenario doesn't crash after the vehicle travels its drive distance
        
        return sequence

    def _create_test_criteria(self):
        """
        Collision test criteria
        """
        # if self.criteria_enable:
        #     return [CollisionTest(self.ego_vehicles[0])]
        return []

    def _setup_scenario_trigger(self, config):
        return None

    def _setup_scenario_end(self, config):
        return None
    
    def get_vehicle_location(self, ego_vehicles):
        ego_x, ego_y = ego_vehicles[0].get_location().x, ego_vehicles[0].get_location().y

        lane_options = set([position[0] for position in parking_lane_waypoints_Town04])
        lane_x = min(lane_options, key=lambda x: abs(x - ego_x))
        dest_x = self.destination_loc.x

        close_x = lane_x + self.lane_width / 2

        if dest_x < lane_x: # parking on the right side
            self.miss_offset *= -1.5

        if self.collision_mode == CollisionMode.COLLIDE:
            if dest_x > close_x:
                self.speed *= 0.8
            spawn_y = min(town04_bound["y_max"], self.destination_loc.y + abs(self.destination_loc.y - ego_y))
            actual_offset = abs(spawn_y - self.destination_loc.y)

            self.trigger_distance = actual_offset * 2
            print(self.trigger_distance)

        elif self.collision_mode == CollisionMode.MISS:

            spawn_y = min(town04_bound["y_max"], self.destination_loc.y + abs(self.destination_loc.y - ego_y) - self.miss_offset)
            actual_offset = abs(spawn_y - self.destination_loc.y)

            self.trigger_distance = actual_offset * 2 + self.miss_offset

            print(self.trigger_distance)

        elif self.collision_mode == CollisionMode.STOP_EARLY:
            if dest_x > close_x: ### if in the opposite lane we collide on the swing out before parking
                self.speed *= 0.8
            spawn_y = min(town04_bound["y_max"], self.destination_loc.y + abs(self.destination_loc.y - ego_y))
            actual_offset = abs(spawn_y - self.destination_loc.y)

            self.trigger_distance = actual_offset * 2
            self.drive_distance = actual_offset - 7 ## stop with leeway

        # Distance from spawn to ego's crossing point — used by GMM to set sweep length.
        self.cross_distance_m = float(actual_offset)

        spawn_location = carla.Location(x=close_x, y=spawn_y, z=1.0)
        return carla.Transform(spawn_location, carla.Rotation(yaw=self.yaw))