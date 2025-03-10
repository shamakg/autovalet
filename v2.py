from math import sqrt
from enum import Enum
from typing import Tuple
from queue import Queue
import json
import carla
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from pykalman import KalmanFilter
from hybrid_a_star.hybrid_a_star import hybrid_a_star_planning as hybrid_a_star
from fisheye_camera import FisheyeCamera, EquidistantProjection 
from v2_perception import run_perception_model
from v2_controller import VehiclePIDController

def kmph_to_mps(speed): return speed/3.6
def mps_to_kmph(speed): return speed*3.6

DESTINATION_THRESHOLD = 0.2
REPLAN_THRESHOLD = 2
LOOKAHEAD = 3
TRAJECTORY_EXTENSION = 5
MAX_ACCELERATION = 1
MAX_SPEED = kmph_to_mps(10)
MIN_SPEED = kmph_to_mps(2)
STOP_CONTROL = carla.VehicleControl(brake=1.0)
STAGNATION_HISTORY_LENGTH = 100
STAGNATION_THRESHOLD = 0.1
FAILURE_HISTORY_LENGTH = 200
FAILURE_THRESHOLD = 0.1

class Mode(Enum):
    DRIVING = 0
    PARKED = 1
    FAILED = 2

class Direction(Enum):
    FORWARD = 0
    REVERSE = 1

    def opposite(self):
        return Direction.FORWARD if self == Direction.REVERSE else Direction.REVERSE

class TrajectoryPoint():
    def __init__(self, direction: Direction, x: float, y: float, speed: float, angle: float):
        self.direction = direction
        self.x = x
        self.y = y
        self.speed = speed
        self.angle = angle

    def distance(self, other):
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def offset(self, sign: int = 1):
        return TrajectoryPoint(self.direction, self.x + 1.4055*sign*np.cos(self.angle), self.y + 1.4055*sign*np.sin(self.angle), self.speed, self.angle)

class ObstacleMap():
    def __init__(self, min_x: int, min_y: int, obs: np.array):
        self.min_x = min_x
        self.min_y = min_y
        self.obs = obs
        self.static_obs = obs.copy()
        self.dyn_obs_clusters = {}
        self.dyn_obs_states = {}
        self.dyn_obs_kfs = {}

    def transform_coord(self, x: float, y: float):
        x = int((x - self.min_x) / 0.25)
        y = int((y - self.min_y) / 0.25)
        return x, y

    def inverse_transform_coord(self, x: float, y: float):
        x = x * 0.25 + self.min_x
        y = y * 0.25 + self.min_y
        return x, y

    def inverse_transform_coords(self, coords: np.array):
        return coords * 0.25 + np.array([self.min_x, self.min_y])

    def circular_mask(self, x: float, y: float, r: float):
        x, y = self.transform_coord(x, y)
        r = int(r / 0.25)
        x_coords, y_coords = np.ogrid[:self.obs.shape[0], :self.obs.shape[1]]
        mask = np.sqrt((x_coords - x) ** 2 + (y_coords - y) ** 2) < r
        return mask

    def probs(self):
        return 1 - (1 / (1 + np.exp(self.obs)))
    
    def generate_collision_mask(self, trajectory: TrajectoryPoint | list[TrajectoryPoint]):
        mask = np.zeros_like(self.obs, dtype=bool)
        x_coords, y_coords = np.meshgrid(np.arange(self.obs.shape[0]), np.arange(self.obs.shape[1]))

        for wp in trajectory[:min(10, len(trajectory)-TRAJECTORY_EXTENSION):2] if isinstance(trajectory, list) else [trajectory]:
            x, y = self.transform_coord(wp.x, wp.y)
            R = np.array([
                [np.cos(-wp.angle), -np.sin(-wp.angle)],
                [np.sin(-wp.angle), np.cos(-wp.angle)]
            ])
            coords = np.stack([x_coords.flatten() - x, y_coords.flatten() - y])
            rotated = R @ coords
            rear_distance = 1.045 * 4
            front_distance = 3.856 * 4
            width = 2.18 * 4
            hits = (rotated[0] >= -rear_distance) & (rotated[0] <= front_distance) & (rotated[1] >= -width/2) & (rotated[1] <= width/2)
            mask |= hits.reshape(self.obs.shape[1], self.obs.shape[0]).T

        return mask

    def check_collision(self, trajectory: list[TrajectoryPoint]):
        probs = self.probs()
        mask = self.generate_collision_mask(trajectory)
        # robs = np.zeros_like(probs)
        # robs[mask & (probs > 0.5)] = 1
        # robs = robs[::-1]
        # plt.cla()
        # plt.imshow(robs, cmap='gray', vmin=0, vmax=1)
        # plt.savefig('obs_map2.png')

        # robs = np.zeros_like(probs)
        # robs[mask | (probs > 0.5)] = 1
        # robs = robs[::-1]
        # plt.cla()
        # plt.imshow(robs, cmap='gray', vmin=0, vmax=1)
        # plt.savefig('obs_map3.png')

        return np.any(probs[mask] > 0.5)

def refine_trajectory(trajectory: list[TrajectoryPoint]):
    if len(trajectory) == 0: return

    # find direction changes based on positions
    segments = [0]
    cur_direction = trajectory[0].direction
    forward_vec_x = np.cos(trajectory[0].angle)
    forward_vec_y = np.sin(trajectory[0].angle)
    if cur_direction == Direction.REVERSE:
        forward_vec_x = -forward_vec_x
        forward_vec_y = -forward_vec_y
    for i in range(len(trajectory) - 1):
        dx = trajectory[i+1].x - trajectory[i].x
        dy = trajectory[i+1].y - trajectory[i].y
        dist = sqrt(dx**2 + dy**2)
        if dist == 0:
            continue
        dot = dx * forward_vec_x + dy * forward_vec_y
        forward_vec_x = dx
        forward_vec_y = dy
        if dot < 0:
            cur_direction = trajectory[i].direction = cur_direction.opposite()
            segments.append(i)
        else:
            trajectory[i].direction = cur_direction
    if len(trajectory) > 1:
        trajectory[-1].direction = trajectory[-2].direction
    segments.append(len(trajectory))

    for segment_i in range(len(segments) - 1):
        start = segments[segment_i]
        end = segments[segment_i + 1]

        # forward pass
        for i in range(start + 1, end - 1):
            d = trajectory[i-1].distance(trajectory[i])
            trajectory[i].speed = min(MAX_SPEED, sqrt(trajectory[i-1].speed**2 + 2 * MAX_ACCELERATION * d))

        # backward pass
        for i in range(end - 2, start - 1, -1):
            d = trajectory[i-1].distance(trajectory[i])
            trajectory[i].speed = min(trajectory[i].speed, sqrt(trajectory[i+1].speed**2 + 2 * MAX_ACCELERATION * d))

def plan_hybrid_a_star(cur: TrajectoryPoint, destination: TrajectoryPoint, obs: ObstacleMap) -> list[TrajectoryPoint]:
    # run planner
    start = np.array([cur.x - obs.min_x, cur.y - obs.min_y, cur.angle])
    end = np.array([destination.x - obs.min_x, destination.y - obs.min_y, destination.angle])
    local_min_x = min(cur.x, destination.x) - 6
    local_max_x = max(cur.x, destination.x) + 6
    local_min_y = min(cur.y, destination.y) - 6
    local_max_y = max(cur.y, destination.y) + 6
    local_min_x, local_min_y = obs.transform_coord(local_min_x, local_min_y)
    local_max_x, local_max_y = obs.transform_coord(local_max_x, local_max_y)
    ox = []
    oy = []
    probs = obs.probs()
    probs[0, :] = 1
    probs[-1, :] = 1
    probs[:, 0] = 1
    probs[:, -1] = 1

    probs[local_min_x, :] = 1
    probs[local_max_x, :] = 1
    probs[:, local_min_y] = 1
    probs[:, local_max_y] = 1

    for coord in np.argwhere(probs > 0.5):
        ox.append(coord[0]*.25)
        oy.append(coord[1]*.25)

    hybrid_astar_path = hybrid_a_star(start, end, ox, oy, 2.0, np.deg2rad(15.0))
    if not hybrid_astar_path:
        return []
    result_x = hybrid_astar_path.x_list
    result_y = hybrid_astar_path.y_list
    result_yaw = hybrid_astar_path.yaw_list
    result_direction = hybrid_astar_path.direction_list

    # sometimes the direction list is too short
    if len(result_direction) < len(result_x):
        for _ in range(len(result_x) - len(result_direction)):
            result_direction.append(result_direction[-1])

    # generate trajectory points
    trajectory = [TrajectoryPoint(Direction.FORWARD if d else Direction.REVERSE, x + obs.min_x, y + obs.min_y, MIN_SPEED, yaw) for x, y, yaw, d in zip(result_x, result_y, result_yaw, result_direction)]
    trajectory[0].speed = cur.speed
    trajectory[0].angle = cur.angle
    refine_trajectory(trajectory)
    
    return trajectory

class CarlaTimeSensor():
    def __init__(self, world):
        self.world = world

    def get_time(self):
        return self.world.get_snapshot().timestamp.elapsed_seconds

# TODO: get data from actual GNSS sensor instead of getting
# perfect vehicle data from CARLA
class CarlaGnssSensor():
    def __init__(self, actor):
        self.actor = actor

    def get_location(self) -> Tuple[float, float]:
        loc = self.actor.get_location()
        return loc.x, loc.y

    def get_speed(self) -> float:
        vel = self.actor.get_velocity()
        return vel.length()
    
    def get_heading(self):
        return np.deg2rad(self.actor.get_transform().rotation.yaw)

class CarlaCameraSensor():
    def __init__(self, actor, world):
        self.actor = actor
        self.cameras = {}
        with open('./v2_camera_config.json', 'r') as file:
            calibration = json.load(file)
            for cam_name in calibration:
                cam_config = calibration[cam_name]
                x = cam_config['spawn_point']['x']
                y = -cam_config['spawn_point']['y']
                z = cam_config['spawn_point']['z']
                roll = cam_config['spawn_point']['roll']
                pitch = cam_config['spawn_point']['pitch']
                yaw = cam_config['spawn_point']['yaw']
                quat = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_quat()
                roll, pitch, yaw = Rotation.from_quat(quat).as_euler('xyz', degrees=True)
                pitch = -pitch
                yaw = -yaw
                width = cam_config['x_size']
                height = cam_config['y_size']
                fx = cam_config['f_x']
                fy = cam_config['f_y']
                cx = cam_config['c_x']
                cy = cam_config['c_y']
                max_angle = cam_config['max_angle']
                cam = FisheyeCamera(
                    parent_actor=actor, camera_model=EquidistantProjection, width=width, height=height, tick=0.0,
                    x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, fx=fx, fy=fy, cx=cx, cy=cy,
                    k0=0.0, k1=0.0, k2=0.0, k3=0.0, k4=0.0,
                    max_angle=max_angle, camera_type='sensor.camera.rgb'
                )

                self.cameras[cam_name] = cam

    def get_images(self):
        images = {}
        for cam_name in self.cameras:
            self.cameras[cam_name].create_fisheye_image()
            images[cam_name] = self.cameras[cam_name].image
        return images

    def destroy(self):
        for cam in self.cameras.values():
            cam.destroy()

class CarlaCollisionSensor():
    def __init__(self, actor, world):
        self.actor = actor
        self.world = world
        self.has_collided = False
        # collision_bp = world.get_blueprint_library().find('sensor.other.collision')
        # sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=actor)
        # sensor.listen(self.on_collision)

class CarlaCar():
    def __init__(self, world, blueprint, spawn_point, destination, destination_bb, debug=False):
        self.world = world
        self.actor = world.spawn_actor(blueprint, spawn_point)
        self.time_sensor = CarlaTimeSensor(world)
        self.gnss_sensor = CarlaGnssSensor(self.actor)
        self.camera_sensor = CarlaCameraSensor(self.actor, world)
        self.collision_sensor = CarlaCollisionSensor(self.actor, world)
        self.car = Car((destination.x, destination.y), self.time_sensor, self.gnss_sensor, self.camera_sensor, self.collision_sensor)
        self.destination_bb = destination_bb

        self.debug = debug
        if debug:
            self.debug_init(spawn_point, destination)

    def calculate_critical_time(self): return self.car.calculate_critical_time()
    def localize(self): self.car.localize()
    def perceive(self, cur_x, cur_y, cur_angle, imgs): return self.car.perceive(cur_x, cur_y, cur_angle, imgs)
    def plan(self): self.car.plan()
    def fail(self): self.car.fail()

    def run_step(self):
        self.actor.apply_control(self.car.control())
        if self.debug:
            self.debug_step()
        
    def debug_init(self, spawn_point, destination):
        self.world.debug.draw_string(spawn_point.location, 'start', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=120.0, persistent_lines=True)
        self.world.debug.draw_string(destination, 'end', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=120.0, persistent_lines=True)

    def debug_step(self):
        cur = self.car.cur
        self.world.debug.draw_string(carla.Location(x=cur.x, y=cur.y), 'X', draw_shadow=False, color=carla.Color(r=0, g=255, b=0), life_time=0.1, persistent_lines=True)
        for loc in self.car.trajectory:
            self.world.debug.draw_string(carla.Location(x=loc.x, y=loc.y), 'o', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=1.0, persistent_lines=True)

    def destroy(self):
        self.actor.destroy()
        self.camera_sensor.destroy()

    def iou(self):
        actor = self.actor
        car_transform = actor.get_transform()
        car_loc = car_transform.location
        car_angle = car_transform.rotation.yaw
        car_angle = np.deg2rad(actor.get_transform().rotation.yaw)
        car_rotation = np.array([
            [np.cos(car_angle), -np.sin(car_angle)],
            [np.sin(car_angle), np.cos(car_angle)]
        ])
        car_bb = [
            -actor.bounding_box.extent.x, -actor.bounding_box.extent.y,
            actor.bounding_box.extent.x, actor.bounding_box.extent.y
        ]
        car_vertices = [
            np.dot(car_rotation, np.array([car_bb[0], car_bb[1]])) + np.array([car_loc.x, car_loc.y]),
            np.dot(car_rotation, np.array([car_bb[0], car_bb[3]])) + np.array([car_loc.x, car_loc.y]),
            np.dot(car_rotation, np.array([car_bb[2], car_bb[3]])) + np.array([car_loc.x, car_loc.y]),
            np.dot(car_rotation, np.array([car_bb[2], car_bb[1]])) + np.array([car_loc.x, car_loc.y])
        ]
        destination_bb = self.destination_bb
        destination_vertices = [(destination_bb[0], destination_bb[1]), (destination_bb[0], destination_bb[3]), (destination_bb[2], destination_bb[3]), (destination_bb[2], destination_bb[1])]

        # Debug bounding boxes
        if self.debug:
            self.world.debug.draw_string(carla.Location(x=car_vertices[0][0], y=car_vertices[0][1]), 'o', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=0.1, persistent_lines=True)
            self.world.debug.draw_string(carla.Location(x=car_vertices[1][0], y=car_vertices[1][1]), 'o', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=0.1, persistent_lines=True)
            self.world.debug.draw_string(carla.Location(x=car_vertices[2][0], y=car_vertices[2][1]), 'o', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=0.1, persistent_lines=True)
            self.world.debug.draw_string(carla.Location(x=car_vertices[3][0], y=car_vertices[3][1]), 'o', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=0.1, persistent_lines=True)

            self.world.debug.draw_string(carla.Location(x=destination_vertices[0][0], y=destination_vertices[0][1]), 'o', draw_shadow=False, color=carla.Color(r=0, g=255, b=0), life_time=0.1, persistent_lines=True)
            self.world.debug.draw_string(carla.Location(x=destination_vertices[1][0], y=destination_vertices[1][1]), 'o', draw_shadow=False, color=carla.Color(r=0, g=255, b=0), life_time=0.1, persistent_lines=True)
            self.world.debug.draw_string(carla.Location(x=destination_vertices[2][0], y=destination_vertices[2][1]), 'o', draw_shadow=False, color=carla.Color(r=0, g=255, b=0), life_time=0.1, persistent_lines=True)
            self.world.debug.draw_string(carla.Location(x=destination_vertices[3][0], y=destination_vertices[3][1]), 'o', draw_shadow=False, color=carla.Color(r=0, g=255, b=0), life_time=0.1, persistent_lines=True)

        car_polygon = Polygon(car_vertices)
        destination_polygon = Polygon(destination_vertices)
        iou = car_polygon.intersection(destination_polygon).area / car_polygon.union(destination_polygon).area
        return iou

class Car():
    def __init__(self, destination: Tuple[float, float], time_sensor: CarlaTimeSensor, gnss_sensor: CarlaGnssSensor, camera_sensor: CarlaCameraSensor, collision_sensor: CarlaCollisionSensor):
        self.cur = TrajectoryPoint(Direction.FORWARD, 0, 0, 0, 0)
        self.stagnation_history = []
        self.failure_history = []
        self.obs: ObstacleMap = []
        self.destination = TrajectoryPoint(Direction.FORWARD, destination[0], destination[1], MIN_SPEED, 0).offset(-1)
        self.controller = VehiclePIDController({'K_P': 2, 'K_I': 0.05, 'K_D': 0.2, 'dt': 0.05}, {'K_P': 0.5, 'K_I': 0.05, 'K_D': 0.0, 'dt': 0.05})
        self.time_sensor = time_sensor
        self.gnss_sensor = gnss_sensor
        self.camera_sensor = camera_sensor
        self.collision_sensor = collision_sensor
        self.trajectory: list[TrajectoryPoint] = []
        self.ti = 0
        self.mode = Mode.DRIVING
        self.time = 0
    
    def calculate_critical_time(self):
        probs = self.obs.probs()
        collision_mask = self.obs.generate_collision_mask(self.trajectory[self.ti:])
        cur_x = self.cur.x
        cur_y = self.cur.y
        inner_radius = 3
        inner_mask = self.obs.circular_mask(cur_x, cur_y, inner_radius)
        uncertain_coords = self.obs.inverse_transform_coords(np.argwhere(collision_mask & ~inner_mask & (0.4 <= probs) & (probs <= 0.6)))
        # recording_obs = np.zeros_like(probs)
        # recording_obs[collision_mask & ~inner_mask & (0.4 <= probs) & (probs <= 0.6)] = 1
        # recording_obs = recording_obs[::-1]
        # plt.cla()
        # plt.imshow(recording_obs, cmap='gray', vmin=0, vmax=1)
        # plt.savefig('obs_map2.png')
        if len(uncertain_coords) == 0:
            return float('inf')
        min_distance_to_uncertain = np.min(np.linalg.norm(uncertain_coords - np.array([cur_x, cur_y]), axis=1))
        time_to_uncertain = (min_distance_to_uncertain + inner_radius) / self.cur.speed
        stopping_time = self.cur.speed / MAX_ACCELERATION
        return time_to_uncertain - stopping_time

    def localize(self):
        point = np.array([self.cur.x, self.cur.y])
        self.stagnation_history.append(point)
        self.failure_history.append(point)
        if len(self.stagnation_history) > STAGNATION_HISTORY_LENGTH:
            self.stagnation_history.pop(0)
        if len(self.failure_history) > FAILURE_HISTORY_LENGTH:
            self.failure_history.pop(0)
        self.time = self.time_sensor.get_time()
        self.cur.x, self.cur.y = self.gnss_sensor.get_location()
        self.cur.speed = self.gnss_sensor.get_speed()
        self.cur.angle = self.gnss_sensor.get_heading()
        self.cur = self.cur.offset(-1)

    def perceive(self, cur_x, cur_y, cur_angle, imgs):
        if self.collision_sensor.has_collided:
            self.mode = Mode.FAILED
            return None, None
        if any(img is None for img in imgs.values()):
            return None, None
        imgnps = [
            imgs['rgb_front'],
            imgs['rgb_left'],
            imgs['rgb_rear'],
            imgs['rgb_right']
        ]
        occ = run_perception_model(cur_x, -cur_y, -cur_angle, imgnps)
        if occ is not None:
            # static updates
            updates = np.zeros_like(self.obs.obs)
            dyn_points = []
            radius = 9
            inner_radius = 3
            mask = self.obs.circular_mask(cur_x, cur_y, radius)
            inner_mask = self.obs.circular_mask(cur_x, cur_y, inner_radius)
            mask &= ~inner_mask
            for point in occ:
                reflected_point = np.array([point[0], -point[1]])
                rotation_matrix = np.array([
                    [np.cos(cur_angle), -np.sin(cur_angle)],
                    [np.sin(cur_angle), np.cos(cur_angle)]
                ])
                rotated_point = np.dot(rotation_matrix, reflected_point)
                if np.linalg.norm(rotated_point) > radius:
                    continue
                absolute_point = rotated_point + np.array([cur_x, cur_y])
                x, y = self.obs.transform_coord(absolute_point[0], absolute_point[1])
                if 0 <= x < len(updates) and 0 <= y < len(updates[0]):
                    if point[-1] == 9: # vehicle
                        updates[x][y] += np.log(0.55 / (1 - 0.55))
                    elif point[-1] == 8: # dynamic obstacle
                        dyn_points.append(absolute_point)
                    else:
                        updates[x][y] += np.log(0.55 / (1 - 0.55))
            for x, y in zip(*np.where(mask)):
                cell_point = np.array(self.obs.inverse_transform_coord(x, y)) - np.array([cur_x, cur_y])
                distance = np.linalg.norm(cell_point)
                distance_weight = max(0, 1 - (distance / radius))
                if updates[x][y] == 0:
                    updates[x][y] = np.log(0.45 / (1 - 0.45))
                updates[x][y] *= distance_weight
            self.obs.static_obs += updates

            # dynamic updates
            dyn_points = np.array(dyn_points)
            if len(dyn_points) > 0:
                clustering = DBSCAN(eps=1.0, min_samples=3).fit(dyn_points)
                clustering_labels = clustering.labels_
                unique_clustering_labels = set(clustering_labels) - {-1}
                cluster_points_list = []
                cluster_centroids = []
                for label in unique_clustering_labels:
                    cluster_points = dyn_points[clustering_labels == label]
                    cluster_points_list.append(cluster_points)
                    cluster_centroids.append(np.mean(cluster_points, axis=0))
                prev_ids = list(self.obs.dyn_obs_clusters.keys())
                prev_cluster_centroids = [self.obs.dyn_obs_clusters[prev_id][0] for prev_id in prev_ids]
                obj_assignments = {}
                if len(cluster_centroids) > 0 and len(prev_cluster_centroids) > 0:
                    distances = cdist(cluster_centroids, prev_cluster_centroids)
                    for i in range(len(cluster_centroids)):
                        min_dist_i = np.argmin(distances[i])
                        if distances[i, min_dist_i] < 1.0:
                            obj_id = prev_ids[min_dist_i]
                            obj_assignments[obj_id] = i
                            dt = self.time - self.obs.dyn_obs_clusters[obj_id][2]
                            self.obs.dyn_obs_clusters[obj_id] = (cluster_centroids[i], cluster_points_list[i], self.time)
                            filtered_state_mean, filtered_state_covariance = self.obs.dyn_obs_states[obj_id]
                            self.obs.dyn_obs_states[obj_id] = self.obs.dyn_obs_kfs[obj_id].filter_update(
                                filtered_state_mean=filtered_state_mean,
                                filtered_state_covariance=filtered_state_covariance,
                                observation=cluster_centroids[i],
                                transition_matrix=np.array([
                                    [1, 0, dt, 0],
                                    [0, 1, 0, dt],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]
                                ]),
                                transition_offset=np.array([0, 0, 0, 0]),
                                observation_offset=np.array([0, 0])
                            )
                        else:
                            obj_id = len(self.obs.dyn_obs_clusters)
                            obj_assignments[obj_id] = i
                            self.obs.dyn_obs_clusters[obj_id] = (cluster_centroids[i], cluster_points_list[i], self.time)
                            initial_state_mean, initial_state_covariance = self.obs.dyn_obs_states[obj_id] = (
                                np.array([cluster_centroids[i][0], cluster_centroids[i][1], 0, 0]),
                                1.0 * np.eye(4)
                            )
                            self.obs.dyn_obs_kfs[obj_id] = KalmanFilter(
                                transition_matrices=np.array([
                                    [1, 0, 2, 0],
                                    [0, 1, 0, 2],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]
                                ]),
                                observation_matrices=np.array([
                                    [1, 0, 0, 0],
                                    [0, 1, 0, 0]
                                ]),
                                transition_covariance=0.1 * np.eye(4),
                                observation_covariance=0.05 * np.eye(2),
                                initial_state_mean=initial_state_mean,
                                initial_state_covariance=initial_state_covariance
                            )

                    # for prev_id in prev_ids:
                    #     last_time = self.obs.dyn_obs_history[prev_id][-1][1]
                    #     if self.time - last_time > 5.0:
                    #         del self.obs.dyn_obs_history[prev_id]
                    #         del self.obs.dyn_obs_kfs[prev_id]
                else:
                    for i in range(len(cluster_centroids)):
                        obj_id = len(self.obs.dyn_obs_clusters)
                        obj_assignments[obj_id] = i
                        # self.obs.dyn_obs_history[obj_id] = [(cluster_centroids[i], self.time)]
                        self.obs.dyn_obs_clusters[obj_id] = (cluster_centroids[i], cluster_points_list[i], self.time)
                        initial_state_mean, initial_state_covariance = self.obs.dyn_obs_states[obj_id] = (
                            np.array([cluster_centroids[i][0], cluster_centroids[i][1], 0, 0]),
                            1.0 * np.eye(4)
                        )
                        self.obs.dyn_obs_kfs[obj_id] = KalmanFilter(
                            transition_matrices=np.array([
                                [1, 0, 2, 0],
                                [0, 1, 0, 2],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]
                            ]),
                            observation_matrices=np.array([
                                [1, 0, 0, 0],
                                [0, 1, 0, 0]
                            ]),
                            transition_covariance=0.1 * np.eye(4),
                            observation_covariance=0.05 * np.eye(2),
                            initial_state_mean=initial_state_mean,
                            initial_state_covariance=initial_state_covariance
                        )

            dyn_obs = np.zeros_like(self.obs.obs)
            obj_ids_to_delete = []
            for obj_id in self.obs.dyn_obs_clusters:
                state_mean, _ = self.obs.dyn_obs_states[obj_id]
                original_centroid = self.obs.dyn_obs_clusters[obj_id][0]
                original_time = self.obs.dyn_obs_clusters[obj_id][2]
                if self.time - original_time > 2.0:
                    obj_ids_to_delete.append(obj_id)
                    continue
                for _ in range(int((self.time - original_time)*5 + 10)):
                    state_mean = np.dot(np.array([
                        [1, 0, 0.2, 0],
                        [0, 1, 0, 0.2],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                    ]), state_mean)
                    for point in self.obs.dyn_obs_clusters[obj_id][1]:
                        x, y = self.obs.transform_coord(point[0] - original_centroid[0] + state_mean[0], point[1] - original_centroid[1] + state_mean[1])
                        if 0 <= x < self.obs.obs.shape[0] and 0 <= y < self.obs.obs.shape[1]:
                            self.obs.static_obs[x, y] = 0
                            dyn_obs[x, y] = np.log(0.75 / (1 - 0.75))

            for obj_id in obj_ids_to_delete:
                del self.obs.dyn_obs_clusters[obj_id]
                del self.obs.dyn_obs_states[obj_id]
                del self.obs.dyn_obs_kfs[obj_id]
            
            # combine static and dynamic obstacles
            dyn_obs[self.obs.generate_collision_mask(TrajectoryPoint(Direction.FORWARD, cur_x, cur_y, 0, cur_angle))] = 0
            self.obs.obs = self.obs.static_obs + dyn_obs

                # visualize obj_assignments, which maps obj_id to cluster index
                # for obj_id, kf in self.obs.dyn_obs_states.items():
                #     state = kf[0]
                #     print(f"Object {obj_id}: Pos=[{state[0]:.2f}, {state[1]:.2f}], Vel=[{state[2]:.2f}, {state[3]:.2f}]")
                # plt.cla()
                # colors = plt.cm.get_cmap("tab10", len(cluster_centroids))
                # for obj_id, cluster_i in obj_assignments.items():
                #     cluster_points = cluster_points_list[cluster_i]
                #     color = colors(cluster_i)
                #     plt.scatter(cluster_points[:, 1], cluster_points[:, 0], color=color, label=f'obj{obj_id}')
                #     plt.text(cluster_centroids[cluster_i][1], cluster_centroids[cluster_i][0], f'obj{obj_id}', color=color)
                # for obj_id, kf in self.obs.dyn_obs_states.items():
                #     state = kf[0]
                #     trajectory = [state[:2]]
                #     for _ in range(5):
                #         state = np.dot(np.array([
                #             [1, 0, 0.1, 0],
                #             [0, 1, 0, 0.1],
                #             [0, 0, 1, 0],
                #             [0, 0, 0, 1]
                #         ]), state)
                #         trajectory.append(state[:2])
                #     trajectory = np.array(trajectory)
                #     plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', linestyle='-', label=f'Traj {obj_id}')
                # plt.xlim([100, 300])
                # plt.ylim([150, 300])
                # plt.savefig('dyn_occ.png')
        return imgnps, occ

    def plan(self):
        cur = self.cur
        destination = self.destination

        # replan trajectory if needed
        trajectory = self.trajectory
        should_extend = len(trajectory) == 0
        should_fix = len(trajectory) > 0 and cur.distance(trajectory[self.ti]) > REPLAN_THRESHOLD
        has_collision = self.obs.check_collision(trajectory[self.ti:])
        has_stagnated = len(self.stagnation_history) == STAGNATION_HISTORY_LENGTH and np.linalg.norm(np.mean(self.stagnation_history, axis=0) - np.array([cur.x, cur.y])) < STAGNATION_THRESHOLD
        has_failed = len(self.failure_history) == FAILURE_HISTORY_LENGTH and np.linalg.norm(np.mean(self.failure_history, axis=0) - np.array([cur.x, cur.y])) < FAILURE_THRESHOLD

        if has_failed:
            self.ti = 0
            self.trajectory = []
            self.mode = Mode.FAILED
            return

        if should_extend or should_fix or has_collision or has_stagnated:
            if has_stagnated: print('stagnated')
            new_trajectory = plan_hybrid_a_star(cur, destination, self.obs)

            if not new_trajectory:
                destination.angle += np.pi
                destination = self.destination = destination.offset(-2)
                new_trajectory = plan_hybrid_a_star(cur, destination, self.obs)

            if new_trajectory:
                for i in range(1, TRAJECTORY_EXTENSION+1):
                    new_trajectory.append(destination.offset(i/3))
                self.ti = 1
                trajectory = self.trajectory = new_trajectory
                self.stagnation_history = []
            else:
                self.obs.obs[1:-1, 1:-1] = 0
                self.ti = 0
                self.trajectory = []
                self.plan()
                # self.mode = Mode.FAILED
        
        # decay all obstacles except the edges
        # self.obs.obs[1:-1, 1:-1] *= 0.99
    
    def control(self):
        if self.mode == Mode.FAILED: return STOP_CONTROL

        # stop if close to destination
        cur = self.cur
        destination = self.destination
        distance_to_destination = cur.distance(destination)
        if self.mode == Mode.PARKED or distance_to_destination < DESTINATION_THRESHOLD and self.ti >= len(self.trajectory) - TRAJECTORY_EXTENSION - 1:
            self.mode = Mode.PARKED
            return STOP_CONTROL

        # find next waypoint
        ti = self.ti
        trajectory = self.trajectory
        wp = trajectory[ti]
        wp_dist = cur.distance(wp)
        for i in range(ti + 1, len(trajectory)):
            if cur.distance(trajectory[i]) > wp_dist:
                break
            ti = i
            wp = trajectory[i]
            wp_dist = cur.distance(wp)
        self.ti = ti

        # find lookahead waypoint
        wp = trajectory[ti]
        future_wp = wp
        for i in range(ti + 1, ti + LOOKAHEAD + 1):
            if i >= len(trajectory):
                break
            new_dist = cur.distance(trajectory[i])
            if new_dist < wp_dist:
                break
            future_wp = trajectory[i]
            wp_dist = new_dist

        cur.direction = wp.direction
        ctrl = self.controller.run_step(
            mps_to_kmph(cur.speed),
            mps_to_kmph(wp.speed),
            cur,
            future_wp,
            wp.direction == Direction.REVERSE
        )
        return ctrl

    def run_step(self):
        self.perceive()
        self.plan()
        return self.control()

    def fail(self):
        self.ti = 0
        self.trajectory = []
        self.mode = Mode.FAILED