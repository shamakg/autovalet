from testbed.v2_experiment_utils import (
    load_client,
    is_done,
    town04_load,
    town04_spectator_bev,
    town04_spawn_ego_vehicle,
    town04_spawn_parked_cars,
    town04_spectator_follow,
    town04_get_drivable_graph
)

# For lane waypoint hack
from parking_position import (
    parking_lane_waypoints_Town04
)

import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio

# SCENARIOS = [
#     (17, [16, 18]),
#     (18, [17, 19]),
#     (19, [18, 20]),
#     (20, [19, 21]),
#     (21, [20, 22]),
#     # (22, [21, 23]),
#     # (23, [22, 24]),
#     (24, [23, 25]),
#     (25, [24, 26]),
#     # (26, [25, 27]),
#     (27, [26, 28]),
#     (28, [27, 29]),
#     (29, [28, 30]),
#     # (30, [29, 31]),
#     (31, [30, 32]),
#     # (32, [31, 33]),
#     # (33, [32, 34]),
#     (34, [33, 35]),
#     # (35, [34, 36]),
#     (36, [35, 37]),
#     (37, [36, 38]),
#     # (38, [37, 39]),
#     (39, [38, 40]),
#     (40, [39, 41]),
#     # (41, [40, 42]),
#     # (42, [41, 43]),
#     (43, [42, 44]),
#     # (44, [43, 45]),
#     # (45, [44, 46]),
#     (46, [45, 47]),
#     (47, [46, 48]),
# ]

SCENARIOS = [
    (16, [15, 17]),  # row2 col1
    (17, [16, 18]),  # row2 col2
    (18, [17, 19]),  # row2 col3
    (19, [18, 20]),  # row2 col4
    (20, [19, 21]),  # row2 col5
    (21, [20, 22]),  # row2 col6
    (22, [21, 23]),  # row2 col7
    (23, [22, 24]),  # row2 col8
    (24, [23, 25]),  # row2 col9
    (25, [24, 26]),  # row2 col10
    (26, [25, 27]),  # row2 col11
    (29, [28, 30]),  # row2 col14 ← far hint
    (32, [31, 33]),  # row3 col1
    (33, [32, 34]),  # row3 col2
    (34, [33, 35]),  # row3 col3
    (35, [34, 36]),  # row3 col4
    (36, [35, 37]),  # row3 col5
    (37, [36, 38]),  # row3 col6
    (38, [37, 39]),  # row3 col7
    (39, [38, 40]),  # row3 col8
    (40, [39, 41]),  # row3 col9
    (41, [40, 42]),  # row3 col10
    (42, [41, 43]),  # row3 col11
    (45, [44, 46]),  # row3 col14 ← far hint
]

SCENARIOS = [
    (27, [26, 28]),  # row2 col12
    (28, [27, 29]),  # row2 col13
    (30, [29, 31]),  # row2 col15
    (31, [30, 32]),  # row2 col16
    (43, [42, 44]),  # row3 col12
    (44, [43, 45]),  # row3 col13
    (46, [45, 47]),  # row3 col15
    (47, [46, 48]),  # row3 col16
]
# SCENARIOS = [
#     ### These first two are included in training data, but let's see them
#     # (17, [16, 18]),
#     # (18, [17, 19]),
#     (19, [18, 20]), #dont keep
#     (20, [19, 21]), #dont keep
#     # (21, [20, 22]),
#     (22, [21, 23]), #keep
#     (23, [22, 24]), #keep
#     # (24, [23, 25]),
#     # (25, [24, 26]),
#     (26, [25, 27]),
#     # (27, [26, 28]),
#     # (28, [27, 29]),
#     # (29, [28, 30]),
#     (30, [29, 31]),
#     # (31, [30, 32]),
#     (32, [31, 33]),
#     (33, [32, 34]),
#     # (34, [33, 35]),
#     (35, [34, 36]),
#     # (36, [35, 37]),
#     # (37, [36, 38]),
#     # (38, [37, 39]),
#     # (39, [38, 40]),
#     # (40, [39, 41]),
#     (41, [40, 42]),
#     (42, [41, 43]),
#     (43, [42, 44]),
#     (44, [43, 45]),
#     (45, [44, 46]),
#     # (46, [45, 47]),
#     # (47, [46, 48]),
# ]
NUM_RANDOM_CARS = 25

def run_scenario(world, destination_parking_spot, parked_spots, ious, recording_file):
    try:
        # load parked cars
        parked_cars, parked_cars_bbs = town04_spawn_parked_cars(world, parked_spots, destination_parking_spot, NUM_RANDOM_CARS)

        # load car
        car = town04_spawn_ego_vehicle(world, destination_parking_spot)
        recording_cam = car.init_recording(recording_file)

        # HACK: enable perfect perception of parked cars
        car.car.obs = parked_cars_bbs

        # HACK: set lane waypoints to guide parking in adjacent lanes
        car.car.lane_wps = parking_lane_waypoints_Town04

        # run simulation
        i = 0
        while not is_done(car):
            world.tick()
            car.run_step()
            car.process_recording_frames()
            # town04_spectator_follow(world, car)

        iou = car.iou()
        ious.append(iou)
        print(f'IOU: {iou}')
    finally:
        recording_cam.destroy()
        car.destroy()
        for parked_car in parked_cars:
            parked_car.destroy()

def main():
    try:
        client = load_client()

        # load map
        world = town04_load(client)

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