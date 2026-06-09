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

from parking_position import (
    parking_lane_waypoints_Town04
)
from parking_position import (
    parking_vehicle_locations_Town04,
)

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



class ParkingConeScenario(BasicScenario):
    
    """
    This class holds everything required for a scenario in which another vehicle parked at the side lane
    opens the door, forcing the ego to lane change, invading the opposite lane
    """
    def __init__(self, world, ego_vehicles, config, destination_spot, parked_spots, debug_mode=False, criteria_enable=True,
                 timeout=180):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()

        self.timeout = timeout
        self.destination_spot = destination_spot
        self.parked_spots = parked_spots
        self.cones = []

        self.route_mode = False

        super().__init__("ParkingCone", ego_vehicles, config, world, debug_mode, criteria_enable=criteria_enable)


    def _initialize_actors(self, config):
        """
        Creates a parked vehicle on the side of the road
        """
        CONE_BB_HALF_H = 0.1
        destination_loc = parking_vehicle_locations_Town04[self.destination_spot]
        empty_spots = []

        vehicles_nearby = self._world.get_actors().filter('vehicle.*')

        for spot_index in range(len(parking_vehicle_locations_Town04)):
            if spot_index == self.destination_spot or spot_index in self.parked_spots:
                continue

            spot_loc = parking_vehicle_locations_Town04[spot_index]

            if spot_loc.y < destination_loc.y:
                spot_occupied = False

                for vehicle in vehicles_nearby:
                    v_loc = vehicle.get_location()
                    bb = vehicle.bounding_box
                    # AABB check: is the cone position inside this vehicle's footprint?
                    if (abs(spot_loc.x - v_loc.x) < bb.extent.x + 0.3 and
                            abs(spot_loc.y - v_loc.y) < bb.extent.y + 0.3):
                        spot_occupied = True
                        break

                if not spot_occupied:
                    distance = abs(spot_index - self.destination_spot)
                    empty_spots.append((spot_index, spot_loc, distance))
            
        empty_spots.sort(key=lambda x: x[2])

        cone_bp = self._world.get_blueprint_library().find('static.prop.constructioncone')
        num_cones_to_spawn = min(10, len(empty_spots))

        for i in range(num_cones_to_spawn):
            spot_index, spot_loc, _ = empty_spots[i]

            # Probe terrain z so cones sit on the ground rather than floating at z=1.5.
            # Static props don't respond to gravity, so physics snap via world.tick() doesn't work.
            # Spawn directly at final z (terrain_z + bb_half_h) to avoid creating a stale
            # broadphase entry: spawning at high z then calling set_location() without a tick
            # leaves a ghost collision at the spawn height that can block nearby actor spawns.
            probe = carla.Location(x=spot_loc.x, y=spot_loc.y, z=10.0)
            proj = self._world.ground_projection(probe, 20.0)
            terrain_z = proj.location.z if proj else 0.15
            # static.prop.constructioncone BB half-height is ~0.3 m (measured from bounding_box.extent.z)
            
            place_z = terrain_z + CONE_BB_HALF_H

            ### Two cones per parking spot, spawned directly at ground level
            cone_transform = carla.Transform(
                carla.Location(x=spot_loc.x, y=spot_loc.y - 0.5, z=place_z),
                carla.Rotation(yaw=0)
            )
            cone_transform_2 = carla.Transform(
                carla.Location(x=spot_loc.x, y=spot_loc.y + 0.5, z=place_z),
                carla.Rotation(yaw=0)
            )

            try:
                cone = self._world.spawn_actor(cone_bp, cone_transform)
                cone_2 = self._world.spawn_actor(cone_bp, cone_transform_2)
                if cone and cone_2:
                    cone.set_simulate_physics(False)
                    cone_2.set_simulate_physics(False)
                    # Register after placement so cleanup() always destroys them
                    self.cones.extend([cone, cone_2])
                    self.other_actors.extend([cone, cone_2])
                    print(f"Spawned cone in parking spot {spot_index} at {cone.get_location()}")
                else:
                    print(f"WARNING: Failed to spawn cone in spot {spot_index}")
            except Exception as e:
                print(f"WARNING: Could not spawn cone in spot {spot_index}: {e}")
        

        if len(self.cones) == 0:
            print("No cones spawned.")

    def _setup_scenario_trigger(self, config):
        return None

    def _create_behavior(self):
        """
        Leave space in front, as the TM doesn't detect open doors, and change the opposite frequency 
        so that the ego can pass
        """
        root = py_trees.composites.Sequence(name="ParkingCones")
        root.add_child(WaitForever())

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = [ScenarioTimeoutTest(self.ego_vehicles[0], self.config.name)]
        if not self.route_mode:
            criteria.append(CollisionTest(self.ego_vehicles[0]))
        return criteria

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        self.remove_all_actors()
