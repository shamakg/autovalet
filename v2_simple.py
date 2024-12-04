from math import sqrt
from enum import Enum
from typing import Tuple, List
from queue import Queue
import carla

import numpy as np
import matplotlib.pyplot as plt
from shapely import Polygon
import cv2
from hybrid_a_star.hybrid_a_star import hybrid_a_star_planning as hybrid_a_star
from v2_controller import VehiclePIDController
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

def kmph_to_mps(speed): return speed/3.6
def mps_to_kmph(speed): return speed*3.6

DESTINATION_THRESHOLD = 0.2
REPLAN_THRESHOLD = 2
LOOKAHEAD = 3
TRAJECTORY_EXTENSION = 5
MAX_ACCELERATION = 1
MAX_SPEED = kmph_to_mps(10)
MIN_SPEED = kmph_to_mps(2)
STAGNATION_HISTORY_LENGTH = 100
STAGNATION_THRESHOLD = 0.1
FAILURE_HISTORY_LENGTH = 200
FAILURE_THRESHOLD = 0.1
COLLISION_REPLAN_DEBOUNCE = 1.0  # seconds between collision-triggered replans

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

    def generate_collision_mask(self, trajectory: TrajectoryPoint | list[TrajectoryPoint],
                                front_m=3.856, rear_m=1.045, half_width_m=1.09):
        mask = np.zeros_like(self.obs, dtype=bool)
        x_coords, y_coords = np.meshgrid(np.arange(self.obs.shape[0]), np.arange(self.obs.shape[1]))

        rear_distance  = rear_m      / 0.25
        front_distance = front_m     / 0.25
        width          = half_width_m * 2 / 0.25

        for wp in trajectory[:min(10, len(trajectory)-TRAJECTORY_EXTENSION):2] if isinstance(trajectory, list) else [trajectory]:
            x, y = self.transform_coord(wp.x, wp.y)
            R = np.array([
                [np.cos(-wp.angle), -np.sin(-wp.angle)],
                [np.sin(-wp.angle), np.cos(-wp.angle)]
            ])
            coords = np.stack([x_coords.flatten() - x, y_coords.flatten() - y])
            rotated = R @ coords
            hits = (rotated[0] >= -rear_distance) & (rotated[0] <= front_distance) \
                 & (rotated[1] >= -width/2)       & (rotated[1] <= width/2)
            mask |= hits.reshape(self.obs.shape[1], self.obs.shape[0]).T

        return mask

    def check_collision(self, trajectory: list[TrajectoryPoint],
                        front_m=3.856, rear_m=1.045, half_width_m=1.09):
        probs = self.probs()
        mask = self.generate_collision_mask(trajectory, front_m=front_m, rear_m=rear_m, half_width_m=half_width_m)
        return np.any(probs[mask] > 0.5)

    def check_static_collision(self, trajectory: list[TrajectoryPoint],
                               front_m=3.856, rear_m=1.045, half_width_m=1.09):
        """Check collision against static obstacles only (ignores dynamic obstacles)."""
        probs = 1 - (1 / (1 + np.exp(self.static_obs)))
        mask = self.generate_collision_mask(trajectory, front_m=front_m, rear_m=rear_m, half_width_m=half_width_m)
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

    ### SHAMAK Edit by claude
    h, w = obs.obs.shape
    local_min_x = max(0, min(local_min_x, h - 1))
    local_max_x = max(0, min(local_max_x, h - 1))
    local_min_y = max(0, min(local_min_y, w - 1))
    local_max_y = max(0, min(local_max_y, w - 1))
    #####

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
        self.vehicle_hit = False
        collision_bp = world.get_blueprint_library().find('sensor.other.collision')
        sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=actor)
        sensor.listen(self._on_collision)
        self._sensor = sensor

    def _on_collision(self, event):
        other = event.other_actor
        tid = other.type_id
        if tid.startswith('vehicle.') or tid.startswith('static.prop'):
            self.vehicle_hit = True
            self.has_collided = True

    def destroy(self):
        if hasattr(self, '_sensor') and self._sensor.is_alive:
            self._sensor.destroy()

class CarlaCar():
    def __init__(self, world, blueprint, spawn_point, destination, destination_bb, debug=False):
        self.world = world
        self.actor = world.spawn_actor(blueprint, spawn_point)
        self.gnss_sensor = CarlaGnssSensor(self.actor)
        self.time_sensor = CarlaTimeSensor(self.world)
        self.collision_sensor = CarlaCollisionSensor(self.actor, self.world)
        self.car = Car((destination.x, destination.y), self.gnss_sensor, self.time_sensor, self.collision_sensor, self.world)
        # Override Car dims with actual actor bounding box.
        # car.cur is 1.6m behind the actor origin (localize calls offset(-1)),
        # so we add/subtract that offset when converting extent → cur-relative distances.
        bb = self.actor.bounding_box
        self.car.front_m      = bb.extent.x + bb.location.x + 1.6
        self.car.rear_m       = max(0.1, bb.extent.x - bb.location.x - 1.6)
        self.car.half_width_m = bb.extent.y
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

    def init_recording(self, recording_path, width=480, height=320, fps=20):
        world = self.world
        actor = self.actor
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(width))
        cam_bp.set_attribute('image_size_y', str(height))
        cam_bp.set_attribute('fov', str(90))
        # local-space offset: 10m behind, 5m up, pitched down 20 degrees
        cam_transform = carla.Transform(
            carla.Location(x=-10, z=5),
            carla.Rotation(pitch=-20)
        )
        cam = world.spawn_actor(cam_bp, cam_transform, attach_to=actor, attachment_type=carla.AttachmentType.Rigid)
        self.recording_writer = cv2.VideoWriter(
            recording_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
        self.recording_width = width
        self.recording_height = height
        cam.listen(lambda image: self.frames.put(image))
        return cam


    def process_recording_frames(self):
        while not self.frames.empty():
            if self.has_recorded_segment and self.car.mode == Mode.PARKED:
                return
            image = self.frames.get()
            self.has_recorded_segment = False
            data = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
            data = data[:, :, :3].copy()  # RGBA -> BGR (CARLA gives BGRA)
            self.recording_writer.write(data)
            if self.car.mode == Mode.PARKED:
                self.has_recorded_segment = True
                for _ in range(15):
                    self.recording_writer.write(data)

    def finalize_recording(self):
        if hasattr(self, 'recording_writer') and self.recording_writer:
            self.recording_writer.release()

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
        self.collision_sensor.destroy()
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

        self.time = 0
        self.static_obs = []
        self.parked_car_ids = set()
        self._last_collision_replan_time = -COLLISION_REPLAN_DEBOUNCE
        # Ego vehicle dimensions (metres); overridden by CarlaCar from actual bounding box
        self.front_m      = 3.856
        self.rear_m       = 1.045
        self.half_width_m = 1.09

    def calculate_critical_time(self):
        probs = self.obs.probs()
        collision_mask = self.obs.generate_collision_mask(
            self.trajectory[self.ti:],
            front_m=self.front_m, rear_m=self.rear_m, half_width_m=self.half_width_m,
        )
        cur_x = self.cur.x
        cur_y = self.cur.y
        inner_radius = 3
        inner_mask = self.obs.circular_mask(cur_x, cur_y, inner_radius)
        uncertain_coords = self.obs.inverse_transform_coords(np.argwhere(collision_mask & ~inner_mask & (0.4 <= probs) & (probs <= 0.6)))
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
            return

        self.parked_car_ids = parked_car_ids

        world_actors = self.world.get_actors()
        dyn_obs = np.zeros_like(self.obs.obs, dtype=float)

        for actor in list(world_actors.filter('vehicle.*')) + list(world_actors.filter('walker.*')):
            if actor.id == self.gnss_sensor.actor.id or actor.id in parked_car_ids:
                continue
            transform = actor.get_transform()
            loc = transform.location
            bb = actor.bounding_box
            yaw = np.deg2rad(transform.rotation.yaw)
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)
            for dl in np.arange(-bb.extent.x, bb.extent.x + 0.25, 0.25):
                for dw in np.arange(-bb.extent.y, bb.extent.y + 0.25, 0.25):
                    wx = loc.x + dl * cos_y - dw * sin_y
                    wy = loc.y + dl * sin_y + dw * cos_y
                    xi, yi = self.obs.transform_coord(wx, wy)
                    if 0 <= xi < self.obs.obs.shape[0] and 0 <= yi < self.obs.obs.shape[1]:
                        dyn_obs[xi, yi] = np.log(0.75 / 0.25)

        self.obs.obs = self.obs.static_obs + dyn_obs

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
        trajectory = self.trajectory
        should_extend = len(trajectory) == 0
        should_fix = len(trajectory) > 0 and cur.distance(trajectory[self.ti]) > REPLAN_THRESHOLD
        has_collision = self.obs.check_collision(
            trajectory[self.ti:],
            front_m=self.front_m, rear_m=self.rear_m, half_width_m=self.half_width_m,
        ) if len(trajectory) > 0 else False

        has_stagnated = (len(self.stagnation_history) == STAGNATION_HISTORY_LENGTH and
                         np.linalg.norm(np.mean(self.stagnation_history, axis=0) - np.array([cur.x, cur.y])) < STAGNATION_THRESHOLD)
        has_failed = (len(self.failure_history) == FAILURE_HISTORY_LENGTH and
                      np.linalg.norm(np.mean(self.failure_history, axis=0) - np.array([cur.x, cur.y])) < FAILURE_THRESHOLD)

        if has_failed:
            self.ti = 0
            self.trajectory = []
            self.mode = Mode.FAILED
            return

        collision_replan_allowed = (self.time - self._last_collision_replan_time >= COLLISION_REPLAN_DEBOUNCE)
        effective_has_collision = has_collision and (
            collision_replan_allowed or should_extend or should_fix or has_stagnated)

        if should_extend or should_fix or effective_has_collision or has_stagnated:
            if has_stagnated:
                print('stagnated')
            new_trajectory = plan_hybrid_a_star(cur, destination, self.obs)
            if new_trajectory:
                for i in range(1, TRAJECTORY_EXTENSION + 1):
                    new_trajectory.append(destination.offset(i / 3))
                self.ti = 1
                self.trajectory = new_trajectory
                self.stagnation_history = []
                self.mode = Mode.DRIVING
                self._last_collision_replan_time = self.time
            else:
                self.obs.obs[1:-1, 1:-1] = 0
                self.ti = 0
                self.trajectory = []
                self.mode = Mode.STALLED

        if not has_collision:
            if self.mode == Mode.STALLED and len(self.trajectory) > 0:
                self.mode = Mode.DRIVING

    def run_step(self, parked_car_ids=set()):
        self.perceive(parked_car_ids)
        self.plan()
        return self.control()
