#!/usr/bin/env python
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Pedestrians crossing through the middle of the lane.
"""

from __future__ import print_function

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy,
                                                                      KeepVelocity,
                                                                      WaitForever,
                                                                      Idle,
                                                                      ActorTransformSetter,
                                                                      MovePedestrianWithEgo)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, ScenarioTimeoutTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocation,
                                                                               InTimeToArrivalToLocation,
                                                                               DriveDistance)
from srunner.scenarios.basic_scenario import BasicScenario

from srunner.scenarios.pedestrian_crossing import PedestrianCrossing
from srunner.tools.background_manager import HandleJunctionScenario
import random


from parking_position import (
    parking_lane_waypoints_Town04,
    parking_vehicle_locations_Town04,
)


class SmartPedestrianCrossingParking(BasicScenario):

    """
    This class holds everything required for a group of natual pedestrians crossing the road.
    The ego vehicle is passing through a road,
    And encounters a group of pedestrians crossing the road.

    This is a single ego vehicle scenario.

    Notice that the initial pedestrian will walk from the start of the junction ahead to end_walker_flow_1.
    """

    def __init__(self, world, ego_vehicles, config, parking_spot, min_trigger_dist = 15.0, walker_distance = 10, speed = 1.5, debug_mode=True, criteria_enable=False, timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        ## This class has three random elements: whether pedestrians will cross or not, 
        ## which side a given pedestrian crosses from 
        #3 and whether they will look both ways before crossing

        self._wmap = CarlaDataProvider.get_map()

        self._rng = CarlaDataProvider.get_random_seed()

        self._adversary_speed = 1.3  # Speed of the adversary [m/s]
        self._reaction_time = 3.5  # Time the agent has to react to avoid the collision [s]
        self._min_trigger_dist = min_trigger_dist # Min distance to the collision location that triggers the adversary [m]
        self._ego_end_distance = 40
        self.timeout = timeout
        self._trigger_location = parking_vehicle_locations_Town04[parking_spot]
        self.ego_vehicles = ego_vehicles

        self.look_both_ways = 1 ## stops for 1 seconds before jaywalking
        self.yaw = 180 ## yaw of arbitrary side, the other is 180 - yaw
        self.depth = 5 ## how far in the parked cars to start pedestrians
        self.car_gap = 1.6 ## distance from simulated waypoint to car gap
        self.walker_speed = 1.3
        self._walker_distance = 6 # approximately lane size, after this point physics is turned off to avoid collision with parked cars
        self.clearance = self.car_gap + .3 # car gap with clearance

        self.spawn_locations = self.generate_jaywalker_spawns(ego_vehicles)

        self.world = world
        self.speed = speed
        self._walker_data = []   

        self.route_mode = False

        super().__init__(
            name="PedestrianCrossingParking",
            ego_vehicles=ego_vehicles,
            config=config,
            world=world,
            debug_mode=False,
            criteria_enable=criteria_enable
        )

        self.route_mode = False

    

    def _create_behavior(self):
        root = py_trees.composites.Parallel(
          policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL,
          name="PedestrianCrossingParking")
      
        class MoveWalkerNoPhysics(py_trees.behaviour.Behaviour):
            def __init__(self, actor, distance, speed):
                super().__init__("MoveWalkerNoPhysics")
                self.actor = actor
                self.distance = distance
                self.speed = speed
                self.moved = 0

            def update(self):
                if self.moved >= self.distance:
                    return py_trees.common.Status.SUCCESS

                dt = CarlaDataProvider.get_world().get_settings().fixed_delta_seconds or 0.05
                step = self.speed * dt

                fwd = CarlaDataProvider.get_transform(self.actor).get_forward_vector()
                loc = self.actor.get_location()
                loc.x += fwd.x * step
                loc.y += fwd.y * step
                self.actor.set_location(loc)
                self.moved += step
                

                return py_trees.common.Status.RUNNING
            
        class SetWalkerPhysics(py_trees.behaviour.Behaviour):
            def __init__(self, actor, enable):
                super().__init__(f"SetPhysics_{enable}")
                self.actor = actor
                self.enable = enable
            def update(self):
                self.actor.set_simulate_physics(self.enable)
                self.actor.set_collisions(self.enable)
                return py_trees.common.Status.SUCCESS

        class CheckEgoNotPast(py_trees.behaviour.Behaviour):
            def __init__(self, ego, walker, crossing_duration):
                super().__init__("CheckEgoNotPast")
                self.ego = ego
                self.walker = walker
                self.crossing_duration = crossing_duration

            def update(self):
                ego_loc = self.ego.get_location()
                ego_vel = self.ego.get_velocity()
                walker_loc = self.walker.get_location()

                pred_y = ego_loc.y + ego_vel.y * self.crossing_duration

                vy = ego_vel.y
                if abs(vy) > 0.05:
                    will_be_past = (pred_y - walker_loc.y) * vy > 0
                else:
                    will_be_past = ego_loc.y < walker_loc.y

                if will_be_past:
                    return py_trees.common.Status.FAILURE
                return py_trees.common.Status.SUCCESS

        for i, (walker_actor, walker_data) in enumerate(zip(self.other_actors, self._walker_data)):
            walker_seq = py_trees.composites.Sequence(name="WalkerCrossing")

            walker_seq.add_child(SetWalkerPhysics(walker_actor, False))
            walker_seq.add_child(ActorTransformSetter(walker_actor, walker_data['transform'], False))

            walker_trigger = py_trees.composites.Parallel(
                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
            walker_trigger.add_child(InTriggerDistanceToLocation(
                self.ego_vehicles[0], walker_data['transform'].location, self._min_trigger_dist))

            walker_seq.add_child(walker_trigger)
            
            
            walker_seq.add_child(Idle(walker_data.get('idle_time')))
            walker_seq.add_child(SetWalkerPhysics(walker_actor, False))
            walker_seq.add_child(MoveWalkerNoPhysics(walker_actor, walker_data['off_distance'], walker_data['speed']))
            walker_seq.add_child(SetWalkerPhysics(walker_actor, True))
            
            ### Randomly choose which pedestrians walk
            if (random.randint(0,1) == 1):
                walker_seq.add_child(WaitForever())
                root.add_child(walker_seq)
                continue

            ### Look both ways
            walker_seq.add_child(Idle(random.randint(0, self.look_both_ways)))

            ### Gate: only cross if ego won't be past this y-position by crossing end
            crossing_gate = py_trees.composites.Selector(name="CrossingGate")
            crossing_seq = py_trees.composites.Sequence(name="CrossIfSafe")
            crossing_seq.add_child(CheckEgoNotPast(
                self.ego_vehicles[0], walker_actor, walker_data['duration']))
            ### Cross lane
            crossing_seq.add_child(KeepVelocity(
                walker_actor, walker_data['speed'], False,
                walker_data['duration'], walker_data['distance']))
            crossing_seq.add_child(SetWalkerPhysics(walker_actor, False))
            ### Move into parked cars on other side
            crossing_seq.add_child(MoveWalkerNoPhysics(
                walker_actor, walker_data['distance'], walker_data['speed']))
            crossing_gate.add_child(crossing_seq)
            crossing_gate.add_child(WaitForever())  # ego already past — stay put
            walker_seq.add_child(crossing_gate)

            walker_seq.add_child(WaitForever())
            root.add_child(walker_seq)

        return root

    def _initialize_actors(self, config):
        ### Overrode this method to ensure that pedestrians can be easily spawned

        # Spawn the walkers
        i = 0
        for start_location in self.spawn_locations:
            spawn_transform = carla.Transform(start_location[0], carla.Rotation(yaw=start_location[1])) ## got from PedestrianCrossing init
            walker = CarlaDataProvider.request_new_actor('walker.*', spawn_transform)
            if walker is None: 
                for walker in self.other_actors:
                    try:
                        if walker and walker.is_alive:
                            walker.destroy()
                    except:
                        pass
                print("Failed to spawn an adversary, skipping")
                continue
            
            walker.set_location(spawn_transform.location + carla.Location(z=-50))
            walker = self._replace_walker(walker, spawn_transform)

            
            self.other_actors.append(walker)
            walker_d = {'x':start_location[0].x, 'y':start_location[0].y, 'z':start_location[0].z, "yaw":start_location[1]} #hard coded yaw
            
            walker_d['transform'] = spawn_transform
            walker_d['distance'] = self._walker_distance
            walker_d['off_distance'] = self.clearance
            walker_d['speed'] = start_location[2]
            walker_d['duration'] = walker_d['distance'] / walker_d['speed']
            walker_d['idle_time'] = 0.3 + (i * 0.3)

            self._walker_data.append(walker_d)
            i += 1
            
    ## Following 3 functions copied from Pedestrian Scenario
    def _setup_scenario_trigger(self, config):
        """Normal scenario trigger but in parallel, a behavior that ensures the pedestrian stays active"""
        trigger_tree = super()._setup_scenario_trigger(config)

        if not self.route_mode:
            return trigger_tree

        parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="ScenarioTrigger")

        for i, walker in enumerate(reversed(self.other_actors)):
            parallel.add_child(MovePedestrianWithEgo(self.ego_vehicles[0], walker, 100))

        parallel.add_child(trigger_tree)
        return parallel

    def _create_test_criteria(self):
        if self.criteria_enable:
            criteria = [ScenarioTimeoutTest(self.ego_vehicles[0], self.config.name), CollisionTest(self.ego_vehicles[0])]
        return criteria
    
    def _replace_walker(self, walker, spawn_transform):
        """As the adversary is probably, replace it with another one"""
        type_id = walker.type_id
        try:
            if walker.is_alive:
                walker.destroy()
        except:
            pass
        spawn_transform.location.z = 0.5
        walker = CarlaDataProvider.request_new_actor(type_id, spawn_transform)
        if not walker:
            raise ValueError("Couldn't spawn the walker substitute")
        walker.set_simulate_physics(False)
        walker.set_location(spawn_transform.location)
        return walker
    
    def build_scenarios(self, ego_vehicle, debug=False):
        """Called periodically by ScenarioManager - we already built scenarios in __init__"""
        pass
    
    def spawn_parked_vehicles(self, ego_vehicle):
        """Called periodically by ScenarioManager - we already spawned in __init__"""
        pass

    def generate_jaywalker_spawns(self, ego_vehicles):
        spawn_data = []

        walker_speed = self.walker_speed

        lane_centers = sorted(set(wp[0] for wp in parking_lane_waypoints_Town04))

        closest_lane_x = min(lane_centers, key=lambda x: abs(x - ego_vehicles[0].get_location().x ))

        lane_waypoints = [
            wp for wp in parking_lane_waypoints_Town04
            if wp[0] == closest_lane_x
        ]

        for lane_wp in lane_waypoints:
            if lane_wp[1] > self._trigger_location.y + 5:
                continue

            lane_x, lane_y = lane_wp

            # randomly decide which side to start
            direction = random.choice([-1, 1])

            ## may decide to remove this, but this ensures the pedestrians don't clog the parking spot
            if abs(lane_wp[1] - self._trigger_location.y) < 4:
                direction = max([-1, 1], key=lambda x:abs(self.ego_vehicles[0].get_location().x -  (lane_x + x * self.depth)))
        

            start_x = lane_x + direction * self.depth
            start_y = lane_y + self.car_gap
            yaw = self.yaw if direction > 0 else 180 - self.yaw   # face across lane

            location = carla.Location(x=start_x, y=start_y, z=0.5)

            spawn_data.append((location, yaw, walker_speed))

        return spawn_data
