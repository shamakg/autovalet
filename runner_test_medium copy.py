### Version 3 of the code, with the goal being to renew the pedestrian crossing abstraction

from leaderboard.autovalet.parking_scenarios.parking_scenario import ParkingScenario
from leaderboard.autovalet.parking_scenarios.parking_scenario_easy import ParkingScenarioEasy

from leaderboard.autovalet.parking_scenarios.parking_scenario_medium import ParkingScenarioMedium
from leaderboard.autovalet.parking_scenarios.parking_scenario_hard import HardMode, ParkingScenarioHard
from leaderboard.autovalet.parking_scenarios.opposite_vehicle_parking import CollisionMode
from leaderboard.autovalet.v2 import ObstacleMap
from srunner.scenariomanager.timer import GameTime
from srunner.scenarioconfigs.scenario_configuration import ActorConfigurationData, ScenarioConfiguration
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

import py_trees
from v2_experiment_utils import (
    get_obstacle_map_from_scenario,
    load_client,
    is_done,
    obstacle_map_from_bbs,
    town04_load,
    town04_spectator_bev,
    update_walker_bbs,
    get_bounding_boxes

)

from parking_position import (
    parking_lane_waypoints_Town04,
    parking_vehicle_locations_Town04,
)


import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio



from v2_experiment import SCENARIOS
NUM_RANDOM_CARS = 25



def run_scenario(world, destination_parking_spot, parked_spots, ious, locations, accelerations, collisions_ref, recording_file):
    recording_cam = None
    parking_scenario = None
    try:  

        config = ScenarioConfiguration()
        config.name = "Parking"
        config.type = "Parking"
        config.town = "Town04_Opt"
        config.other_actors = None
        parking_scenario = ParkingScenarioHard(
            world=world,
            config=config,
            destination=destination_parking_spot,
            parked=parked_spots,
            debug_mode=0,
            criteria_enable=True,
            mode = HardMode.PedMode
        )

        obstacles = get_obstacle_map_from_scenario(parking_scenario)
        print("obs type:", type(parking_scenario.car.car.obs))
        if obstacles:
            # HACK: enable perfect perception of parked cars
            parking_scenario.car.car.obs = obstacles
        else:
            parking_scenario.car.car.obs = ObstacleMap(0, 0, np.zeros((100, 100)))

        #  # HACK: enable perfect perception of parked cars
        # self.car.car.obs = self.parked_cars_bbs

        # HACK: set lane waypoints to guide parking in adjacent lanes
        parking_scenario.car.car.lane_wps = parking_lane_waypoints_Town04
        ### Recording Logic
        # self.record()
        # world.tick()
        

        # parking_scenario.car.run_step()


        recording_cam = parking_scenario.car.init_recording(recording_file)

        world.tick()

        parked_cars_bbs, traffic_cone_bbs, walker_bbs = get_bounding_boxes(parking_scenario)

        has_collided = False

        
        while not is_done(parking_scenario.car):

            world.tick()

            walker_bbs = update_walker_bbs(world)
            
            timestamp = world.get_snapshot().timestamp
            GameTime.on_carla_tick(timestamp)

            CarlaDataProvider.on_carla_tick() ### NEED THIS so that trigger distance will work

            location = parking_scenario.car.actor.get_location()
            locations.append(np.array([location.x, location.y]))
            
            acceleration = parking_scenario.car.actor.get_acceleration()
            accelerations.append(np.array([acceleration.x, acceleration.y]))

            car = parking_scenario.car
            collision_mask = car.car.obs.generate_collision_mask(car.car.cur)
            ground_truth_obs = obstacle_map_from_bbs(
                        parked_cars_bbs + traffic_cone_bbs + walker_bbs,
                        car.car.obs
                    ).obs
            if np.any(collision_mask & (ground_truth_obs == 1)):
                if not has_collided:
                    collisions_ref[0] += 1
                has_collided = True
            else:
                has_collided = False

            parking_scenario.car.run_step()

            if parking_scenario.scenario_tree:
                parking_scenario.scenario_tree.tick_once()


            parking_scenario.car.process_recording_frames()    

            if parking_scenario.scenario_tree.status != py_trees.common.Status.RUNNING:
                print(f"Parking Timeout: Scenario Failure")
                break
            
        
        iou = parking_scenario.car.iou()
        ious.append(iou)
        print(f'IOU: {iou}')
        print(f'Manual collision count: {collisions_ref[0]}')
    finally:
        criteria_had_collision, scenario_result = analyze_scenario(parking_scenario)
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

        result, had_collision = "SUCCESS", False
        collision_count = 0
        print("EVENTS:")

        # HACK: Manually terminate each criteria       
        for criterion in parking_scenario.get_criteria():
            criterion.terminate(None)
            print(criterion.name, criterion.test_status)
            if criterion.test_status != "SUCCESS":
                result = "SCENARIO FAILURE"
            if "collision" in criterion.name.lower() and criterion.test_status != "SUCCESS":
                had_collision = True
                print(f"Criteria detected collision!")

        print("FINAL SCENARIO RESULT:", result)
        return had_collision, result
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

        ious = []
        location_lists = []
        acceleration_lists = []
        collisions = [] 

        # run scenarios
        for destination_parking_spot, parked_spots in SCENARIOS:
            print(f'Running scenario: destination={destination_parking_spot}, parked_spots={parked_spots}')

            locations = []
            accelerations = []
            collisions_ref = [0] 

            run_scenario(world, destination_parking_spot, parked_spots, ious, 
                        locations, accelerations, collisions_ref, recording_file)
            
            location_lists.append(locations)
            acceleration_lists.append(accelerations)
            collisions.append(collisions_ref[0])



        print("\n" + "="*60)
        print("=== FINAL STATISTICS ===")
        print("="*60)

        if len(ious) > 0:
            avg_iou = np.mean(ious)
            min_iou = np.min(ious)
            max_iou = np.max(ious)
            print(f"\nIOU Statistics:")
            print(f"  Average IOU: {avg_iou:.3f}")
            print(f"  Min IOU: {min_iou:.3f}")
            print(f"  Max IOU: {max_iou:.3f}")
        
        # Location/acceleration statistics
        print(f"\nData Collection:")
        print(f"  Location samples per scenario: {[len(locs) for locs in location_lists]}")
        print(f"  Acceleration samples per scenario: {[len(accs) for accs in acceleration_lists]}")

        
        # Collision statistics (matching old code output style)
        total_collisions = sum(collisions)
        scenarios_with_collisions = sum(1 for c in collisions if c > 0)
        
        print(f"\nCollision Statistics:")
        print(f"  Total collision events: {total_collisions}")
        print(f"  Scenarios with collisions: {scenarios_with_collisions}/{len(collisions)}")
        if len(collisions) > 0:
            print(f"  Average collisions per scenario: {total_collisions/len(collisions):.2f}")
        print(f"  Collision counts per scenario: {collisions}")

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