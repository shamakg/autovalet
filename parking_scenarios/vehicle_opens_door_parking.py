#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Scenarios in which another (opposite) vehicle 'illegally' takes
priority, e.g. by running a red traffic light.
"""

from __future__ import print_function

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, ScenarioTimeoutTest

from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy,
                                                                      OpenVehicleDoor,
                                                                      SwitchWrongDirectionTest,
                                                                      ScenarioTimeout,
                                                                      Idle,
                                                                      OppositeActorFlow, WaitForever)
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocation,
                                                                               InTimeToArrivalToLocation,
                                                                               DriveDistance,
                                                                               WaitUntilInFrontPosition)
from srunner.scenarios.basic_scenario import BasicScenario

from srunner.tools.background_manager import LeaveSpaceInFront, ChangeOppositeBehavior, StopBackVehicles, StartBackVehicles


def get_value_parameter(config, name, p_type, default):
    if name in config.other_parameters:
        return p_type(config.other_parameters[name]['value'])
    else:
        return default


def get_interval_parameter(config, name, p_type, default):
    if name in config.other_parameters:
        return [
            p_type(config.other_parameters[name]['from']),
            p_type(config.other_parameters[name]['to'])
        ]
    else:
        return default
    


class VehicleOpensDoorTwoWaysParking(BasicScenario):
    """
    This class holds everything required for a scenario in which another vehicle parked at the side lane
    opens the door, forcing the ego to lane change, invading the opposite lane
    """
    def __init__(self, world, ego_vehicles, parked_vehicle, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180, trigger_distance = 6):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()

        self.timeout = timeout
        self._min_trigger_dist = trigger_distance ### hardcoded for now
        self._reaction_time = 5.0

        self._opposite_wait_duration = 5
        self._end_distance = 50

        self._parked_distance = get_value_parameter(config, 'distance', float, 50)
        self._direction = get_value_parameter(config, 'direction', str, 'right')
        if self._direction not in ('left', 'right'):
            raise ValueError(f"'direction' must be either 'right' or 'left' but {self._direction} was given")

        self._max_speed = get_value_parameter(config, 'speed', float, 60)
        self._scenario_timeout = 240

        self._opposite_interval = get_interval_parameter(config, 'frequency', float, [20, 100])

        self._parked_actor = parked_vehicle
        self.other_actors = [self._parked_actor]

        self.route_mode = False

        super().__init__("VehicleOpensDoorTwoWaysParking", ego_vehicles, config, world, debug_mode, criteria_enable=criteria_enable)

        # self.route_mode = False

    ## verbatim from VehiclOpensDoor Scenario
    def _move_waypoint_forward(self, wp, distance):
        dist = 0
        next_wp = wp
        while dist < distance:
            next_wps = next_wp.next(1)
            if not next_wps or next_wps[0].is_junction:
                break
            next_wp = next_wps[0]
            dist += 1
        return next_wp
        

    def _initialize_actors(self, config):
        """
        Creates a parked vehicle on the side of the road
        """
        parked_location = self._parked_actor.get_location()
        self._parked_wp = self._map.get_waypoint(parked_location)

        trigger_location = config.trigger_points[0].location
        starting_wp = self._map.get_waypoint(trigger_location)
        front_wps = starting_wp.next(self._parked_distance)
        if len(front_wps) == 0:
            self.front_wps = self._parked_wp
        else:
            self._front_wp = front_wps[0]


        self.parking_slots.append(self._parked_wp.transform.location)

        self._parked_actor.apply_control(carla.VehicleControl(hand_brake=True))
        # Crank up mass so the car is effectively a wall — gentle door brushes
        # that would normally recoil the car and produce sub-threshold impulses
        # now hit an immovable object, making the ego's collision sensor fire reliably.
        _phys = self._parked_actor.get_physics_control()
        _phys.mass = 100000.0
        self._parked_actor.apply_physics_control(_phys)

        self._end_wp = self._move_waypoint_forward(self._front_wp, self._end_distance)

    ### HACK: THIS is needed to prevent the parent class for adding a trigger that stops following behaviors
    def _setup_scenario_trigger(self, config):
        result = super()._setup_scenario_trigger(config)
        return None 
    ### AFTER hours of debugging, above must be included (not completely sure why)

    def _create_behavior(self):
        collision_location = self._parked_actor.get_location()
        print(f"Trigger distance: {self._min_trigger_dist}m")

        root = py_trees.composites.Sequence(name="VehicleOpensDoorTwoWays")
        
        # end_condition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        # end_condition.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))
        # end_condition.add_child(WaitUntilInFrontPosition(self.ego_vehicles[0], self._end_wp.transform, False))
        
        behavior = py_trees.composites.Sequence(name="Main Behavior")
        

        trigger_adversary = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name="TriggerOpenDoor")
        trigger_adversary.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], collision_location, self._min_trigger_dist))
        behavior.add_child(trigger_adversary)

        
        #open both doors
        door_left, door_right = carla.VehicleDoor.FL, carla.VehicleDoor.FR
        behavior.add_child(OpenVehicleDoor(self._parked_actor, door_left))
        behavior.add_child(OpenVehicleDoor(self._parked_actor, door_right))
        behavior.add_child(Idle(self._opposite_wait_duration))
        behavior.add_child(WaitForever())
        
        # end_condition.add_child(behavior)
        root.add_child(behavior)
        
        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = [ScenarioTimeoutTest(self.ego_vehicles[0], self.config.name)]
        return criteria

    # def cleanup(self):
    #     if self._parked_actor:
    #         self._parked_actor.destroy()

    # def __del__(self):
    #     """
    #     Remove all actors and traffic lights upon deletion
    #     """
    #     self.remove_all_actors()
