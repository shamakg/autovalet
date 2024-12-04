
from math import sqrt
from enum import Enum
from typing import Tuple, List
from queue import Queue
import carla
from pykalman import KalmanFilter 

import numpy as np
import matplotlib.pyplot as plt
from shapely import Polygon
import cv2
from hybrid_a_star.hybrid_a_star import hybrid_a_star_planning as hybrid_a_star
from v2_controller import VehiclePIDController
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

def kmph_to_mps(speed): return speed/3.6
def mps_to_kmph(speed): return speed*3.6

DESTINATION_THRESHOLD = 0.2
REPLAN_THRESHOLD = 2
PERCEPTION_LATENCY = 0.5 # seconds
LOOKAHEAD = 3
TRAJECTORY_EXTENSION = 5
MAX_ACCELERATION = 1
MAX_SPEED = kmph_to_mps(10)
MIN_SPEED = kmph_to_mps(2)
STAGNATION_HISTORY_LENGTH = 100
STAGNATION_THRESHOLD = 0.1
FAILURE_HISTORY_LENGTH = 200
FAILURE_THRESHOLD = 0.1
OBSTACLE_WAIT = 0.4 ## tune 

STOP_CONTROL = carla.VehicleControl(brake=1.0)

class Mode(Enum):
    DRIVING = 0
    PARKED = 1
    STALLED = 2
    FAILED = 3


class Direction(Enum):
    FORWARD = 0
    REVERSE = 1

    def opposite(self):
        return Direction.FORWARD if self == Direction.REVERSE else Direction.REVERSE

def plot_trajectory(trajectory):
    x_coords = [p.x for p in trajectory]
    y_coords = [p.y for p in trajectory]
    speeds = [p.speed for p in trajectory]
    directions = [p.direction for p in trajectory]
    
    # Normalize speeds for color mapping
    norm_speeds = [speed / max(speeds) if max(speeds) > 0 else 0 for speed in speeds]
    
    # Create a colormap
    cmap = plt.get_cmap('viridis')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot trajectory points with arrows indicating direction
    for i in range(len(trajectory) - 1):
        ax.plot([x_coords[i], x_coords[i+1]], [y_coords[i], y_coords[i+1]], color=cmap(norm_speeds[i]))
        dx = x_coords[i+1] - x_coords[i]
        dy = y_coords[i+1] - y_coords[i]
        ax.arrow(x_coords[i], y_coords[i], dx, dy, head_width=0.5, head_length=0.5, fc=cmap(norm_speeds[i]), ec=cmap(norm_speeds[i]))
        ax.text(x_coords[i], y_coords[i], f'{directions[i]}', fontsize=8)
    
    # Add colorbar to indicate speed
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(speeds), vmax=max(speeds)))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Speed (m/s)')
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Trajectory with Direction and Speed')
    ax.grid(True)
    plt.savefig('test.png')

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
        return TrajectoryPoint(self.direction, self.x + 1.6*sign*np.cos(self.angle), self.y + 1.6*sign*np.sin(self.angle), self.speed, self.angle)

class ObstacleMap():
    def __init__(self, min_x: int, min_y: int, obs: np.array):
        self.min_x = min_x
        self.min_y = min_y
        self.obs = obs
        self.static_obs = obs.copy()
        self.dyn_obs_clusters = {}
        self.dyn_obs_states = {}
        self.dyn_obs_kfs = {}
        

    def transform_coord(self, x, y):
        x = int((x - self.min_x) / 0.25)
        y = int((y - self.min_y) / 0.25)
        return x, y

    def inverse_transform_coord(self, x, y):
        x = x * 0.25 + self.min_x
        y = y * 0.25 + self.min_y
        return x, y

    def inverse_transform_coords(self, coords: np.array):
        return coords * 0.25 + np.array([self.min_x, self.min_y])

    def circular_mask(self, x, y, r):
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

def obstacle_map_from_bbs(bbs):
    if not bbs:
        return None
    obs_min_x = min(min(b[0], b[2]) for b in bbs) - 10
    obs_max_x = max(max(b[0], b[2]) for b in bbs) + 10
    obs_min_y = min(min(b[1], b[3]) for b in bbs) - 10
    obs_max_y = max(max(b[1], b[3]) for b in bbs) + 10
    obs_list = []
    for b in bbs:
        for x in np.arange(b[0], b[2], .25):
            obs_list.append((x, b[1]))
            obs_list.append((x, b[3]))
        obs_list.append((b[2], b[1]))
        obs_list.append((b[2], b[3]))
        for y in np.arange(b[1], b[3], .25):
            obs_list.append((b[0], y))
            obs_list.append((b[2], y))
        obs_list.append((b[0], b[3]))
        obs_list.append((b[2], b[3]))
    obs_grid = np.zeros((int((obs_max_x - obs_min_x + 1) / .25), int((obs_max_y - obs_min_y + 1) / .25)), dtype=int)
    obs_grid[0, :] = 1
    obs_grid[-1, :] = 1
    obs_grid[:, 0] = 1
    obs_grid[:, -1] = 1
    for x, y in obs_list:
        xi = int((x - obs_min_x) / .25)
        yi = int((y - obs_min_y) / .25)
        if 0 <= xi < obs_grid.shape[0] and 0 <= yi < obs_grid.shape[1]:
            obs_grid[xi, yi] = 1
    return ObstacleMap(obs_min_x, obs_min_y, obs_grid)

def refine_trajectory(trajectory: List[TrajectoryPoint]):
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
    if len(result_direction) < len(result_x):
        for _ in range(len(result_x) - len(result_direction)):
            result_direction.append(result_direction[-1])
    trajectory = [TrajectoryPoint(Direction.FORWARD if d else Direction.REVERSE, x + obs.min_x, y + obs.min_y, MIN_SPEED, yaw) for x, y, yaw, d in zip(result_x, result_y, result_yaw, result_direction)]
    trajectory[0].speed = cur.speed
    trajectory[0].angle = cur.angle
    refine_trajectory(trajectory)
    return trajectory

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
    
class CarlaTimeSensor():
    def __init__(self, world):
        self.world = world

    def get_time(self):
        return self.world.get_snapshot().timestamp.elapsed_seconds

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
        self.gnss_sensor = CarlaGnssSensor(self.actor)
        self.time_sensor = CarlaTimeSensor(self.world)
        self.collision_sensor = CarlaCollisionSensor(self.actor, self.world)
        self.car = Car((destination.x, destination.y), self.gnss_sensor, self.time_sensor, self.collision_sensor, self.world)
        self.destination_bb = destination_bb
        self.recording_file = None
        self.has_recorded_segment = False
        self.frames = Queue()

        self.debug = debug
        if debug:
            self.debug_init(spawn_point, destination)

    ### CHange made here by shamak to actually reutn the control
    def run_step(self, parked_car_ids=set()):
        ret = self.car.run_step(parked_car_ids)
        self.actor.apply_control(ret)

        if self.debug:
            self.debug_step()

        return ret
        
    def init_recording(self, recording_file):
        self.recording_file = recording_file
        world = self.world
        actor = self.actor
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(1080))
        cam_bp.set_attribute('image_size_y', str(720))
        cam_bp.set_attribute('fov', str(90))
        cam_location = actor.get_transform().transform(carla.Location(x=-10, z=5))
        cam_rotation = actor.get_transform().rotation
        cam_rotation.pitch -= 20
        cam_transform = carla.Transform(cam_location, cam_rotation)
        cam = world.spawn_actor(cam_bp, cam_transform, attach_to=actor, attachment_type=carla.AttachmentType.Rigid)
        cam.listen(lambda image: self.frames.put(image))
        return cam
    
    
    def process_recording_frames(self):
        while not self.frames.empty():
            if self.has_recorded_segment and self.car.mode == Mode.PARKED:
                return
            image = self.frames.get()
            self.has_recorded_segment = False
            recording_file = self.recording_file
            data = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
            data = data[:, :, :3].copy()
            data = cv2.putText(
                data,
                "autonomous, 3x speed",
                (20, image.height - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 255, 255),
                thickness=2
            )
            data = cv2.putText(
                data,
                "IOU: {:.2f}".format(self.iou()),
                (image.width - 175, image.height - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 255, 255),
                thickness=2
            )
            recording_file.write_frame(data, pixel_format='bgr24')
            if self.car.mode == Mode.PARKED:
                self.has_recorded_segment = True
                for _ in range(15):
                    recording_file.write_frame(data, pixel_format='bgr24')
    
    def debug_init(self, spawn_point, destination):
        self.world.debug.draw_string(spawn_point.location, 'start', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=120.0, persistent_lines=True)
        self.world.debug.draw_string(destination, 'end', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=120.0, persistent_lines=True)

    def debug_step(self):
        cur = self.car.cur
        self.world.debug.draw_string(carla.Location(x=cur.x, y=cur.y), 'X', draw_shadow=False, color=carla.Color(r=0, g=255, b=0), life_time=0.1, persistent_lines=True)
        if not self.car.trajectory:
            return
        future_wp = self.car.trajectory[min(self.car.ti + LOOKAHEAD, len(self.car.trajectory) - 1)]
        self.world.debug.draw_string(carla.Location(x=future_wp.x, y=future_wp.y), 'o', draw_shadow=False, color=carla.Color(r=0, g=0, b=255), life_time=0.1, persistent_lines=True)
        
        for i, loc in enumerate(self.car.trajectory):
            if i == min(self.car.ti + LOOKAHEAD, len(self.car.trajectory) - 1):
                color = carla.Color(r=0, g=0, b=255)
            else:
                color = carla.Color(r=255, g=0, b=0)
            self.world.debug.draw_string(carla.Location(x=loc.x, y=loc.y), 'o', draw_shadow=False, color=color, life_time=1.0, persistent_lines=True)

    def destroy(self):
        self.actor.destroy()

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
    def __init__(self, destination: Tuple[float, float], gnss_sensor: CarlaGnssSensor, time_sensor: CarlaTimeSensor, collision_sensor: CarlaCollisionSensor, world):
        self.cur = TrajectoryPoint(Direction.FORWARD, 0, 0, 0, 0)
        self.stagnation_history = []
        self.failure_history = []
        self.obs: ObstacleMap | None = None
        self.destination = TrajectoryPoint(Direction.FORWARD, destination[0], destination[1], MIN_SPEED, 0).offset(-1)
        self.controller = VehiclePIDController({'K_P': 2, 'K_I': 0.05, 'K_D': 0.2, 'dt': 0.05}, {'K_P': 0.5, 'K_I': 0.05, 'K_D': 0.0, 'dt': 0.05})
        
        if destination[0] < 284:
            self.destination.angle += np.pi
            self.destination = self.destination.offset(-2)

        self.gnss_sensor = gnss_sensor
        self.time_sensor = time_sensor
        self.collision_sensor = collision_sensor
        self.trajectory: List[TrajectoryPoint] = []
        self.ti = 0
        self.mode = Mode.DRIVING
        self.prev_obs_hash = None
        self.world = world
        

        self.dyn_obs_clusters = {}
        self.dyn_obs_states = {}
        self.dyn_obs_kfs = {}
        self.time = 0
        self.static_obs = []
        self.parked_car_ids = set()
        self.kalman_filters = {}
        self.collision_wait_start = None
        self.has_waited_for_obstacle = False

        self.last_perception_time = -float('inf')

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

    def perceive(self, parked_car_ids=set()):
        if self.collision_sensor.has_collided:
            self.mode = Mode.FAILED
            return None, None
        # self.cur.x, self.cur.y = self.gnss_sensor.get_location()
        # self.cur.speed = self.gnss_sensor.get_speed()
        # self.cur.angle = self.gnss_sensor.get_heading()
        # self.cur = self.cur.offset(-1)

        # self.time = self.time_sensor.get_time()
        self.parked_car_ids = parked_car_ids

        #### only get the real obstacle maps if greater than the perception latency time
        if (self.time - self.last_perception_time) >= PERCEPTION_LATENCY:
            self.ground_truth_kalman_filter(self.world)
            self.last_perception_time = self.time
        else:
            self._predict_only()


    def control(self):
        if self.mode == Mode.STALLED or self.mode == Mode.FAILED: return STOP_CONTROL

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

    def plan(self):
        cur = self.cur
        destination = self.destination
        distance_to_destination = cur.distance(destination)

        trajectory = self.trajectory
        should_extend = len(trajectory) == 0
        should_fix = len(trajectory) > 0 and cur.distance(trajectory[self.ti]) > REPLAN_THRESHOLD
        has_collision = self.obs.check_collision(trajectory[self.ti:]) if len(trajectory) > 0 else False
        has_stagnated = len(self.stagnation_history) == STAGNATION_HISTORY_LENGTH and np.linalg.norm(np.mean(self.stagnation_history, axis=0) - np.array([cur.x, cur.y])) < STAGNATION_THRESHOLD
        has_failed = len(self.failure_history) == FAILURE_HISTORY_LENGTH and np.linalg.norm(np.mean(self.failure_history, axis=0) - np.array([cur.x, cur.y])) < FAILURE_THRESHOLD

        if has_failed:
            self.ti = 0
            self.trajectory = []
            self.mode = Mode.FAILED
            return
        
        if has_collision and not should_extend and not should_fix and not has_stagnated:
            if not self.has_waited_for_obstacle:
                if self.collision_wait_start is None:
                    self.collision_wait_start = self.time
                if self.time - self.collision_wait_start < OBSTACLE_WAIT:
                    self.mode = Mode.STALLED
                    return  
                self.collision_wait_start = None
                self.has_waited_for_obstacle = True
        else:
            if not has_collision:
                self.collision_wait_start = None

        if should_extend or should_fix or has_collision or has_stagnated:
            if has_stagnated: print('stagnated')
            new_trajectory = plan_hybrid_a_star(cur, destination, self.obs)

            # retry with a different angle if the first attempt failed
            # disabling this for now to control experiment better
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
                self.collision_wait_start = None
                self.mode = Mode.DRIVING
            else:
                self.obs.obs[1:-1, 1:-1] = 0
                self.ti = 0
                self.trajectory = []
                # self.plan()
                self.mode = Mode.STALLED
        
        # decay all obstacles except the edges
        # self.obs.obs[1:-1, 1:-1] *= 0.99

    def run_step(self, parked_car_ids=set()):
        self.perceive(parked_car_ids)
        self.plan()
        return self.control()

    def ground_truth_kalman_filter(self, dt = 0.5):

        ### get the obstacles
        world_actors = self.world.get_actors()
        moving_vehicles = world_actors.filter('vehicle.*')
        walkers = world_actors.filter('walker.*')
        obstacles = list(moving_vehicles) + list(walkers)
        ###

        dyn_points = []
        actor_ids = []
        bounding_boxes = []

        ### get the obstacles + their locations
        for obstacle in obstacles:
            if obstacle.id == self.gnss_sensor.actor.id or obstacle.id in self.parked_car_ids:
                continue

            actor_ids.append(obstacle.id)
            loc = obstacle.get_location()
            dyn_points.append([loc.x, loc.y])
            bounding_boxes.append(obstacle.bounding_box)

        ### so now we have an array of the obstacle locations
        dyn_points = np.array(dyn_points)
        dyn_obs = np.zeros_like(self.obs.obs, dtype=float)

        if len(dyn_points) > 0:
            
            ### not entirely sure what this does, could help with noisy perception
            cluster_centroids = [np.array(p) for p in dyn_points]
            cluster_bbs = bounding_boxes
            
            prev_ids = list(self.obs.dyn_obs_clusters.keys())
            prev_cluster_centroids = [self.obs.dyn_obs_clusters[prev_id]['centroid'] for prev_id in prev_ids]
            
            obj_assignments = {}

            if len(cluster_centroids) > 0 and len(prev_cluster_centroids) > 0:
                distances = cdist(cluster_centroids, prev_cluster_centroids)
                for i in range(len(cluster_centroids)):
                    min_dist_i = np.argmin(distances[i])
                    if distances[i, min_dist_i] < 1.0:
                        obj_id = prev_ids[min_dist_i]
                        obj_assignments[obj_id] = i
                        
                        kf_data = self.obs.dyn_obs_clusters[obj_id]
                        dt_actual = self.time - kf_data['last_time']


                        ## run kalman filter on REAL observation
                        filtered_state_mean, filtered_state_covariance = kf_data['kf'].filter_update(
                            filtered_state_mean=kf_data['state_mean'],
                            filtered_state_covariance=kf_data['state_cov'],
                            observation=cluster_centroids[i],
                            transition_matrix=np.array([
                                [1, 0, dt_actual, 0],
                                [0, 1, 0, dt_actual],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]
                            ]),
                            transition_offset=np.array([0, 0, 0, 0]),
                            observation_offset=np.array([0, 0])
                        )

                        points = []
                        for dx in np.arange(-cluster_bbs[i].extent.x, cluster_bbs[i].extent.x, 0.25):
                            for dy in np.arange(-cluster_bbs[i].extent.y, cluster_bbs[i].extent.y, 0.25):
                                points.append([cluster_centroids[i][0] + dx, cluster_centroids[i][1] + dy])

                        
                        self.obs.dyn_obs_clusters[obj_id]['state_mean'] = filtered_state_mean
                        self.obs.dyn_obs_clusters[obj_id]['state_cov'] = filtered_state_covariance
                        self.obs.dyn_obs_clusters[obj_id]['centroid'] = cluster_centroids[i]
                        self.obs.dyn_obs_clusters[obj_id]['last_time'] = self.time
                        self.obs.dyn_obs_clusters[obj_id]['bb'] = cluster_bbs[i]
                        self.obs.dyn_obs_clusters[obj_id]['cluster_points'] = np.array(points)
                        
                    else:
                        obj_id = max(self.obs.dyn_obs_clusters.keys(), default=-1) + 1
                        obj_assignments[obj_id] = i

                        initial_state_mean = np.array([cluster_centroids[i][0], cluster_centroids[i][1], 0, 0])
                        initial_state_covariance = 1.0 * np.eye(4)

                        points = []
                        for dx in np.arange(-cluster_bbs[i].extent.x, cluster_bbs[i].extent.x, 0.25):
                            for dy in np.arange(-cluster_bbs[i].extent.y, cluster_bbs[i].extent.y, 0.25):
                                points.append([cluster_centroids[i][0] + dx, cluster_centroids[i][1] + dy])


                        self.obs.dyn_obs_clusters[obj_id] = {
                            'kf': KalmanFilter(
                                transition_matrices=np.array([
                                    [1, 0, dt, 0],
                                    [0, 1, 0, dt],
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
                            ),
                            'state_mean': initial_state_mean,
                            'state_cov': initial_state_covariance,
                            'centroid': cluster_centroids[i],
                            'last_time': self.time,
                            'bb': cluster_bbs[i],
                            'cluster_points': np.array(points) 
                        }
            else:
                # No previous objects, initialize all as new
                for i in range(len(cluster_centroids)):
                    obj_id = i
                    obj_assignments[obj_id] = i
                    
                    initial_state_mean = np.array([cluster_centroids[i][0], cluster_centroids[i][1], 0, 0])
                    initial_state_covariance = 1.0 * np.eye(4)

                    points = []
                    for dx in np.arange(-cluster_bbs[i].extent.x, cluster_bbs[i].extent.x, 0.25):
                        for dy in np.arange(-cluster_bbs[i].extent.y, cluster_bbs[i].extent.y, 0.25):
                            points.append([cluster_centroids[i][0] + dx, cluster_centroids[i][1] + dy])

                        
                    self.obs.dyn_obs_clusters[obj_id] = {
                        'kf': KalmanFilter(
                            transition_matrices=np.array([
                                [1, 0, dt, 0],
                                [0, 1, 0, dt],
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
                        ),
                        'state_mean': initial_state_mean,
                        'state_cov': initial_state_covariance,
                        'centroid': cluster_centroids[i],
                        'last_time': self.time,
                        'bb': cluster_bbs[i],
                        'cluster_points': np.array(points) 
                    }
            
            obj_ids_to_delete = []
            for obj_id in self.obs.dyn_obs_clusters:
                if self.time - self.obs.dyn_obs_clusters[obj_id]['last_time'] > 2.0:
                    obj_ids_to_delete.append(obj_id)
            
            for obj_id in obj_ids_to_delete:
                del self.obs.dyn_obs_clusters[obj_id]
            

            for obj_id in self.obs.dyn_obs_clusters:
                kf_data = self.obs.dyn_obs_clusters[obj_id]
                state_mean = kf_data['state_mean'].copy()
                original_centroid = kf_data['centroid']
                for _ in range(int((self.time - kf_data['last_time']) * 5 + 10)):
                    state_mean = np.dot(np.array([
                        [1, 0, 0.2, 0],
                        [0, 1, 0, 0.2],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                    ]), state_mean)
                
                predicted_pos = state_mean[:2]
                bb = kf_data['bb']
                uncertainty_margin = 0.3
                
                for point in kf_data['cluster_points']:
                    x, y = self.obs.transform_coord(
                        point[0] - original_centroid[0] + state_mean[0],
                        point[1] - original_centroid[1] + state_mean[1]
                    )
                    if 0 <= x < self.obs.obs.shape[0] and 0 <= y < self.obs.obs.shape[1]:
                        self.obs.static_obs[x, y] = 0
                        dyn_obs[x, y] = np.log(0.75 / (1 - 0.75))
        
        self.obs.obs = self.obs.static_obs + dyn_obs

    ### This is how we predict when we have no new data given to the ego.
    def _predict_only(self):
        dyn_obs = np.zeros_like(self.obs.obs, dtype=float)
        for obj_id in self.obs.dyn_obs_clusters:
            kf_data = self.obs.dyn_obs_clusters[obj_id]
            state_mean = kf_data['state_mean'].copy()
            original_centroid = kf_data['centroid']
            dt_since_last = self.time - kf_data['last_time']
            for _ in range(int(dt_since_last / 0.1) + 10):
                state_mean = np.dot(np.array([
                    [1, 0, 0.1, 0],
                    [0, 1, 0, 0.1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ]), state_mean)
            for point in kf_data['cluster_points']:
                x, y = self.obs.transform_coord(
                    point[0] - original_centroid[0] + kf_data['centroid'][0],  
                    point[1] - original_centroid[1] + kf_data['centroid'][1]
                )
                if 0 <= x < self.obs.obs.shape[0] and 0 <= y < self.obs.obs.shape[1]:
                    dyn_obs[x, y] = np.log(0.75 / (1 - 0.75))
        self.obs.obs = self.obs.static_obs + dyn_obs