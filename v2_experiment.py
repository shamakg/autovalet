from v2_experiment_utils import (
    load_client,
    is_done,
    town04_load,
    town04_spectator_bev,
    town04_spawn_ego_vehicle,
    town04_spawn_parked_cars,
    town04_spawn_traffic_cones,
    town04_spawn_walkers,
    update_walkers,
    obstacle_map_from_bbs,
    clear_obstacle_map,
    clear_destination_obstacle_map,
    init_third_person_camera,
    ms_to_ticks
)

from v2_visualization import (
    Visualizer
)

from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import cv2

SCENARIOS = [
    # (17, [16, 18]),
    # (18, [17, 19]),
    # (19, [18, 20]),
    (20, [19, 21]),
    (21, [20, 22]),
    (22, [21, 23]),
    (23, [22, 24]),
    # (24, [23, 25]),
    # (25, [24, 26]),
    # (26, [25, 27]),
    # (27, [26, 28]),
    # (28, [27, 29]),
    # (29, [28, 30]),
    # (30, [29, 31]),
    # (31, [30, 32]),
    # (32, [31, 33]),
    (33, [32, 34]),
    # (34, [33, 35]),
    (35, [34, 36]),
    (36, [35, 37]),
    (37, [36, 38]),
    # (38, [37, 39]),
    # (39, [38, 40]),
    # (40, [39, 41]),
    # (41, [40, 42]),
    # (42, [41, 43]),
    # (43, [42, 44]),
    # (44, [43, 45]),
    # (45, [44, 46]),
    # (46, [45, 47]),
    # (47, [46, 48]),
]
NUM_RANDOM_CARS = 50
NETWORK_SEND_LATENCIES = [0] # ms
PERCEPTION_LATENCY = 200 # ms
SMALL_PERCEPTION_LATENCY = 100 # ms
RECV_LATENCY = 100 # ms
MODE_PERIOD = 500 # ms
PLANNING_PERIOD = 500 # ms
DATA_COLLECTION_PERIOD = 200 # ms
SHOULD_PIPELINE = True
SHOULD_ADJUST_MODEL = False
TIMEOUT = 120 * 1000 # ms

class Mode(Enum):
    NORMAL = 1
    ALTERNATE = 2 # latency compensation

class EventType(Enum):
    PERCEPTION = 1

class Event:
    def __init__(self, type: EventType, time: int, data):
        self.type = type
        self.time = time
        self.data = data


def run_scenario(client, destination_parking_spot, parked_spots, latency, ious, locations, accelerations, collisions, visualizer):
    try:
        random.seed(9897105114)

        # load map
        world = town04_load(client)

        # load spectator
        town04_spectator_bev(world)

        # load parked cars
        parked_cars, parked_cars_bbs = town04_spawn_parked_cars(world, parked_spots, destination_parking_spot, NUM_RANDOM_CARS)

        # spawn traffic cones
        traffic_cones, traffic_cone_bbs = town04_spawn_traffic_cones(world, [
            (284, -230),
            (287, -225),
        ])

        # spawn walker
        walkers, walker_bbs = town04_spawn_walkers(world, [
            (284, -232),
        ])

        # tick world to load actors
        world.tick()

        # load car
        car = town04_spawn_ego_vehicle(world, destination_parking_spot)

        # load visualization data
        recording_img = None
        def set_recording_img(image):
            nonlocal recording_img
            data = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
            data = data[:, :, :3].copy()
            data = data[:, :, ::-1]
            recording_img = data
        recording_cam = init_third_person_camera(world, car.actor)
        recording_cam.listen(set_recording_img)
        recording_imgnps = None
        recording_occ = None
        recording_obs = None

        # HACK: enable perfect perception of parked cars
        car.car.obs = clear_obstacle_map(obstacle_map_from_bbs(parked_cars_bbs + traffic_cone_bbs + walker_bbs))

        # tick world to load car and cameras
        world.tick()

        # run simulation
        i = 0
        events = []
        recv_latency = 0 if latency == 0 else RECV_LATENCY
        perception_delay = ms_to_ticks(latency + PERCEPTION_LATENCY + recv_latency)
        perception_period = ms_to_ticks(max(latency, PERCEPTION_LATENCY, recv_latency)) if SHOULD_PIPELINE else perception_delay
        small_perception_delay = ms_to_ticks(latency*0.5 + SMALL_PERCEPTION_LATENCY + recv_latency)
        small_perception_period = ms_to_ticks(max(latency*0.5, SMALL_PERCEPTION_LATENCY, recv_latency)) if SHOULD_PIPELINE else small_perception_delay
        mode = Mode.NORMAL
        has_collided = False
        # timeline_diagram = TimelineDiagram()
        while not is_done(car):
            walker_bbs = update_walkers(walkers)
            world.tick()
            car.localize()

            # HACK: manually check for collisions because carla's collision sensor is broken
            collision_mask = car.car.obs.generate_collision_mask(car.car.cur)
            ground_truth_obs = obstacle_map_from_bbs(parked_cars_bbs + traffic_cone_bbs + walker_bbs, car.car.obs).obs
            if np.any(collision_mask & (ground_truth_obs == 1)):
                if not has_collided:
                    collisions[0] += 1
                has_collided = True
            else:
                has_collided = False

            if i % ms_to_ticks(DATA_COLLECTION_PERIOD) == 0:
                location = car.actor.get_location()
                locations.append(np.array([location.x, location.y]))

                acceleration = car.actor.get_acceleration()
                accelerations.append(np.array([acceleration.x, acceleration.y]))

            if SHOULD_ADJUST_MODEL and i % ms_to_ticks(MODE_PERIOD) == 0:
                critical_time = car.calculate_critical_time()
                risk_normal = 1 - (1 if critical_time * 1000 > latency + PERCEPTION_LATENCY + recv_latency else 0) * 0.95
                risk_alternate = 1 - (1 if critical_time * 1000 > latency*0.5 + SMALL_PERCEPTION_LATENCY + recv_latency else 0) * 0.85
                prev_mode = mode

                if risk_normal < risk_alternate:
                    mode = Mode.NORMAL
                else:
                    mode = Mode.ALTERNATE

                if mode != prev_mode:
                    print(f'switching mode: {prev_mode} -> {mode}')

            # print(critical_time * 1000, latency + PERCEPTION_LATENCY + recv_latency)
            # print(f'risk_normal: {risk_normal}, risk_alternate: {risk_alternate}')
            # print()

            if i % (perception_period if mode == Mode.NORMAL else small_perception_period) == 0:
                imgs = car.car.camera_sensor.get_images()
                if mode == Mode.ALTERNATE:
                    for img_name in imgs:
                        img = imgs[img_name]
                        if img is None: break
                        img_shape = img.shape
                        imgs[img_name] = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
                        imgs[img_name] = cv2.resize(img, (img_shape[1], img_shape[0]))

                cur = car.car.cur
                # timeline_diagram.add_send(i, latency)
                # timeline_diagram.add_runtime(i + latency, PERCEPTION_LATENCY)
                # timeline_diagram.add_recv(i + latency + PERCEPTION_LATENCY, recv_latency)
                events.append(Event(EventType.PERCEPTION, i + (perception_delay if mode == Mode.NORMAL else small_perception_delay), (cur.x, cur.y, cur.angle, imgs)))

            for event in events:
                if i != event.time: continue
                if event.type == EventType.PERCEPTION:
                    cur_x, cur_y, cur_angle, imgs = event.data
                    recording_imgnps, recording_occ = car.perceive(cur_x, cur_y, cur_angle, imgs)
                    clear_destination_obstacle_map(car.car.obs, destination_parking_spot)
                    recording_obs = car.car.obs.probs().copy()
                    recording_obs[np.where(obstacle_map_from_bbs(parked_cars_bbs + traffic_cone_bbs + walker_bbs, car.car.obs).obs == 1)] = 1
                    recording_obs = recording_obs[::-1]

            if i % ms_to_ticks(PLANNING_PERIOD) == 0:
                car.plan()

            car.run_step()

            if i > ms_to_ticks(TIMEOUT):
                car.fail()

            visualizer.send(recording_img, recording_imgnps, recording_occ, recording_obs, car.iou(), latency)
            i += 1

        iou = car.iou()
        ious.append(iou)
        print(f'IOU: {iou}')
    finally:
        recording_cam.destroy()
        car.destroy()
        for parked_car in parked_cars:
            parked_car.destroy()
        for traffic_cone in traffic_cones:
            traffic_cone.destroy()
        for walker in walkers:
            walker.destroy()
        world.tick()

def main():
    try:
        client = load_client()

        # load visualizer
        visualizer = Visualizer()

        # run scenarios
        latency_data = []
        for latency in NETWORK_SEND_LATENCIES:
            ious = []
            location_lists = []
            acceleration_lists = []
            collisions = []
            print(f'running scenarios for latency: {latency}ms')
            for destination_parking_spot, parked_spots in SCENARIOS:
                locations = []
                accelerations = []
                collisions_ref = [0]
                print(f'running scenario: destination={destination_parking_spot}, parked_spots={parked_spots}')
                run_scenario(client, destination_parking_spot, parked_spots, latency, ious, locations, accelerations, collisions_ref, visualizer)
                location_lists.append(locations)
                acceleration_lists.append(accelerations)
                collisions.append(collisions_ref[0])
            latency_data.append((latency, ious, location_lists, acceleration_lists, collisions))
            
        # scatter ious for each latency value
        # plt.clf()
        # for latency, ious, accelerations in latency_data:
        #     jerks = np.diff(accelerations)
        #     x_scatter = np.random.normal(loc=latency, scale=0.05, size=len(jerks))
        #     plt.scatter(x_scatter, jerks, alpha=0.6, label=f'{latency}ms')
        # plt.title('Parking IOU Values')
        # plt.xticks(NETWORK_SEND_LATENCIES, [f'{latency}ms' for latency in NETWORK_SEND_LATENCIES])
        # plt.xlabel('Perception Latency')
        # plt.ylabel('IOU Value')
        # plt.grid(True, linestyle='--', alpha=0.5)
        # plt.savefig('iou_scatter.png')

        # graph ious
        # plt.clf()
        # plt.boxplot(ious, positions=[1], vert=True, patch_artist=True, widths=0.5,
        #     boxprops=dict(facecolor='lightblue', color='blue'),
        #     medianprops=dict(color='red'))
        # x_scatter = np.random.normal(loc=0.5, scale=0.05, size=len(ious))
        # plt.scatter(x_scatter, ious, color='darkblue', alpha=0.6, label='Data Points')
        # plt.xticks([1], ['IOU Values'])  # Set x-ticks at the boxplot
        # plt.title('Parking IOU Values')
        # plt.ylabel('IOU Value')
        # plt.grid(True, linestyle='--', alpha=0.5)
        # plt.legend()
        # plt.savefig('iou_boxplot.png')

    except KeyboardInterrupt:
        print('stopping simulation')
    
    finally:
        with open('experiment_data.pkl', 'wb') as f:
            pickle.dump(latency_data, f)
        visualizer.close()

if __name__ == '__main__':
    main()