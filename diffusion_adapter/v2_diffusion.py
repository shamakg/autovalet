"""v2_diffusion.py — v2.py with a _call_planner() abstraction for the diffusion planner.

Keeps the fisheye CarlaCameraSensor on CarlaCar for diffusion model image input.
All shared types/constants/utilities are imported from v2.py to ensure Mode identity
(so is_done() comparisons in v2_experiment_utils work correctly).
"""
import json
import os
from typing import List, Tuple
from queue import Queue

import carla
import numpy as np
import cv2
from shapely import Polygon
from scipy.spatial.transform import Rotation
from fisheye_camera import FisheyeCamera, EquidistantProjection

from v2_controller import VehiclePIDController
from v2 import (
    Mode, Direction, TrajectoryPoint, ObstacleMap,
    CarlaGnssSensor, CarlaTimeSensor, CarlaCollisionSensor,
    obstacle_map_from_bbs, refine_trajectory, plan_hybrid_a_star,
    mps_to_kmph,
    DESTINATION_THRESHOLD, REPLAN_THRESHOLD,
    LOOKAHEAD, TRAJECTORY_EXTENSION, MAX_ACCELERATION, MAX_SPEED, MIN_SPEED,
    STAGNATION_HISTORY_LENGTH, STAGNATION_THRESHOLD,
    FAILURE_HISTORY_LENGTH, FAILURE_THRESHOLD,
    COLLISION_REPLAN_DEBOUNCE,
    POSITION_NOISE_STD,
    STOP_CONTROL,
)

from diffusion_adapter.adapter import diffusion_plan
from utils.map_process import _map_processor
from utils.coord_utils import standard_to_carla

# ---------------------------------------------------------------------------
# Diffusion planner flag — set to True to enable, False to use Hybrid A* only
# ---------------------------------------------------------------------------
USE_DIFFUSION = True
REPLAN_INTERVAL = 1  # replan every tick as the original paper does https://arxiv.org/pdf/2501.15564
DIFFUSION_LOOKAHEAD = 5  # lookahead steps for pure-pursuit (overrides v2.LOOKAHEAD=3 for the 80-step diffusion trajectory)
LOOKAHEAD_DIST_M = 3.0
LOOKAHEAD = 3
DESTINATION_THRESHOLD = 0.5


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


class CarlaCar():
    def __init__(self, world, blueprint, spawn_point, destination, destination_bb, debug=False):
        self.world = world
        self.actor = world.spawn_actor(blueprint, spawn_point)
        self.gnss_sensor = CarlaGnssSensor(self.actor)
        self.time_sensor = CarlaTimeSensor(self.world)
        self.collision_sensor = CarlaCollisionSensor(self.actor, self.world)
        self.camera_sensor = CarlaCameraSensor(self.actor, self.world)
        self.car = Car((destination.x, destination.y), self.gnss_sensor, self.time_sensor, self.collision_sensor, self.world)
        # Override Car dims with actual actor bounding box.
        bb = self.actor.bounding_box
        self.car.front_m      = bb.extent.x + bb.location.x
        self.car.rear_m       = max(0.1, bb.extent.x - bb.location.x)
        self.car.half_width_m = bb.extent.y
        self.destination_bb = destination_bb
        self.recording_file = None
        self.has_recorded_segment = False
        self.frames = Queue()

        self.debug = debug
        if debug:
            self.debug_init(spawn_point, destination)

    def run_step(self, parked_car_ids=set()):
        ret = self.car.run_step(parked_car_ids)
        self.actor.apply_control(ret)

        if self.debug:
            self.debug_step()

        return ret

    def init_recording(self, recording_path, width=480, height=320, fps=20, top_down=False):
        world = self.world
        actor = self.actor
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(width))
        cam_bp.set_attribute('image_size_y', str(height))
        cam_bp.set_attribute('fov', str(90))
        if top_down:
            cam_transform = carla.Transform(
                carla.Location(x=0, z=30),
                carla.Rotation(pitch=-90)
            )
        else:
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
        self.recording_fov = 90
        self.recording_path = recording_path
        self.recording_cam = cam
        cam.listen(lambda image: self.frames.put(image))
        return cam

    def _world_to_pixels(self, image, points_3d):
        """Project 3D world points to 2D pixel coords using the camera's transform."""
        if not points_3d:
            return []
        cam_transform = image.transform
        world_to_cam = np.array(cam_transform.get_inverse_matrix())
        fov_rad = np.deg2rad(self.recording_fov)
        focal = self.recording_width / (2.0 * np.tan(fov_rad / 2.0))
        cx = self.recording_width / 2.0
        cy = self.recording_height / 2.0

        pixels = []
        for wx, wy, wz in points_3d:
            cam_pt = world_to_cam @ np.array([wx, wy, wz, 1.0])
            x, y, z = cam_pt[0], cam_pt[1], cam_pt[2]
            if x <= 0:
                pixels.append(None)
                continue
            u = int(focal * y / x + cx)
            v = int(-focal * z / x + cy)
            if 0 <= u < self.recording_width and 0 <= v < self.recording_height:
                pixels.append((u, v))
            else:
                pixels.append(None)
        return pixels

    def process_recording_frames(self):
        while not self.frames.empty():
            if self.has_recorded_segment and self.car.mode == Mode.PARKED:
                return
            image = self.frames.get()
            self.has_recorded_segment = False
            data = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
            data = data[:, :, :3].copy()  # RGBA -> BGR (CARLA gives BGRA)

            # Overlay trajectory points
            diffusion_pts = getattr(self.car, '_overlay_diffusion', [])
            astar_pts = getattr(self.car, '_overlay_astar', [])
            for px in self._world_to_pixels(image, diffusion_pts):
                if px:
                    cv2.circle(data, px, 4, (0, 255, 0), -1)
            # for px in self._world_to_pixels(image, astar_pts):
            #     if px:
            #         cv2.circle(data, px, 3, (0, 255, 255), -1)

            self.recording_writer.write(data)
            if self.car.mode == Mode.PARKED:
                self.has_recorded_segment = True
                for _ in range(15):
                    self.recording_writer.write(data)

    def finalize_recording(self):
        if hasattr(self, 'recording_writer') and self.recording_writer:
            self.recording_writer.release()
            path = getattr(self, 'recording_path', None)
            if path and os.path.exists(path):
                tmp = path + '.tmp.mp4'
                os.rename(path, tmp)
                os.system(f'ffmpeg -y -i "{tmp}" -vcodec libx264 -crf 18 -acodec aac "{path}" -loglevel quiet')
                os.remove(tmp)

    def debug_init(self, spawn_point, destination):
        self.world.debug.draw_string(spawn_point.location, 'start', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=120.0, persistent_lines=True)
        self.world.debug.draw_string(destination, 'end', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=120.0, persistent_lines=True)

    def debug_step(self):
        cur = self.car.cur
        self.world.debug.draw_string(carla.Location(x=cur.x, y=cur.y), 'X', draw_shadow=False, color=carla.Color(r=0, g=255, b=0), life_time=0.1, persistent_lines=True)
        if not self.car.trajectory:
            return
        future_wp = self.car.trajectory[min(self.car.ti + DIFFUSION_LOOKAHEAD, len(self.car.trajectory) - 1)]
        self.world.debug.draw_string(carla.Location(x=future_wp.x, y=future_wp.y), 'o', draw_shadow=False, color=carla.Color(r=0, g=0, b=255), life_time=0.1, persistent_lines=True)
        for i, loc in enumerate(self.car.trajectory):
            if i == min(self.car.ti + DIFFUSION_LOOKAHEAD, len(self.car.trajectory) - 1):
                color = carla.Color(r=0, g=0, b=255)
            else:
                color = carla.Color(r=255, g=0, b=0)
            self.world.debug.draw_string(carla.Location(x=loc.x, y=loc.y), 'o', draw_shadow=False, color=color, life_time=1.0, persistent_lines=True)

    def destroy(self):
        self.collision_sensor.destroy()
        self.camera_sensor.destroy()
        self.actor.destroy()

    def iou(self):
        actor = self.actor
        car_transform = actor.get_transform()
        car_loc = car_transform.location
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
        angle = np.pi if destination[0] < 284 else 0.0
        self.destination = TrajectoryPoint(Direction.FORWARD, destination[0], destination[1], MIN_SPEED, angle)
        self.controller = VehiclePIDController({'K_P': 2, 'K_I': 0.05, 'K_D': 0.2, 'dt': 0.05}, {'K_P': 0.5, 'K_I': 0.05, 'K_D': 0.0, 'dt': 0.05})

        self.gnss_sensor = gnss_sensor
        self.time_sensor = time_sensor
        self.collision_sensor = collision_sensor
        self.trajectory: List[TrajectoryPoint] = []
        self.ti = 0
        self.mode = Mode.DRIVING
        self.prev_obs_hash = None
        self.world = world

        self.dyn_obs_clusters = {}
        self.time = 0
        self.static_obs = []
        self.parked_car_ids = set()

        self._last_collision_replan_time = -COLLISION_REPLAN_DEBOUNCE
        self._tick = 0
        # Ego vehicle dimensions (metres); overridden by CarlaCar from actual bounding box
        self.front_m      = 3.856
        self.rear_m       = 1.045
        self.half_width_m = 1.09

    def _call_planner(self, cur: TrajectoryPoint, destination: TrajectoryPoint, obs: ObstacleMap) -> list:
        if USE_DIFFUSION:
            return diffusion_plan(cur, destination, obs, self.world)
        return plan_hybrid_a_star(cur, destination, obs)

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

    def perceive(self, parked_car_ids=set()):
        if self.collision_sensor.has_collided:
            self.mode = Mode.FAILED
            return
        self.parked_car_ids = parked_car_ids

        world_actors = self.world.get_actors()
        obstacles = list(world_actors.filter('vehicle.*')) + list(world_actors.filter('walker.*'))

        dyn_obs = np.zeros_like(self.obs.obs, dtype=float)
        new_clusters = {}
        LOG_ODDS = np.log(0.75 / 0.25)

        for obstacle in obstacles:
            if obstacle.id == self.gnss_sensor.actor.id:
                continue
            transform = obstacle.get_transform()
            loc = transform.location
            noisy_x = loc.x + np.random.normal(0, POSITION_NOISE_STD)
            noisy_y = loc.y + np.random.normal(0, POSITION_NOISE_STD)
            bb = obstacle.bounding_box
            yaw = np.deg2rad(transform.rotation.yaw)

            cos_y, sin_y = np.cos(yaw), np.sin(yaw)
            for dl in np.arange(-bb.extent.x, bb.extent.x, 0.25):
                for dw in np.arange(-bb.extent.y, bb.extent.y, 0.25):
                    wx = noisy_x + dl * cos_y - dw * sin_y
                    wy = noisy_y + dl * sin_y + dw * cos_y
                    gx, gy = self.obs.transform_coord(wx, wy)
                    if 0 <= gx < self.obs.obs.shape[0] and 0 <= gy < self.obs.obs.shape[1]:
                        dyn_obs[gx, gy] = LOG_ODDS

            ## TODO: add noise to velocity
            if obstacle.id in parked_car_ids:
                vx, vy = 0.0, 0.0
            else:
                vel = obstacle.get_velocity()
                vx, vy = vel.x, vel.y
            actor_type = 'pedestrian' if 'walker' in obstacle.type_id else 'vehicle'
            new_clusters[obstacle.id] = {
                'state_mean': np.array([noisy_x, noisy_y, vx, vy]),
                'bb': bb,
                'yaw': yaw,   # radians, from transform.rotation.yaw — already computed
                'actor_type': actor_type,
            }

        self.obs.dyn_obs_clusters = new_clusters
        self.obs.obs = self.obs.static_obs + dyn_obs

        print(f"Total obstacles detected: {len(obstacles)}")
        print(f"dyn_obs_clusters size: {len(self.obs.dyn_obs_clusters)}")

    def control(self):
        if self.mode == Mode.STALLED or self.mode == Mode.FAILED: return STOP_CONTROL

        cur = self.cur
        destination = self.destination
        distance_to_destination = cur.distance(destination)
        ti = self.ti
        trajectory = self.trajectory

        last_wp = trajectory[-1] if trajectory else None
        def _remaining_path_length(trajectory, ti):
            total = 0.0
            for i in range(ti, len(trajectory) - 1):
                total += trajectory[i].distance(trajectory[i+1])
            return total
        print("Remaining arc length ", _remaining_path_length(trajectory, ti))
        print("DISTANCE TO DEST:", distance_to_destination)
        near_end_of_trajectory = last_wp is not None and cur.distance(last_wp) < 0.5
            

        if self.mode == Mode.PARKED or (distance_to_destination < DESTINATION_THRESHOLD): #and _remaining_path_length(trajectory, ti) < 3.0):
            self.mode = Mode.PARKED
            return STOP_CONTROL

        
        wp = trajectory[ti]
        wp_dist = cur.distance(wp)
        for i in range(ti + 1, len(trajectory)):
            if cur.distance(trajectory[i]) > wp_dist:
                break
            ti = i
            wp = trajectory[i]
            wp_dist = cur.distance(wp)
        self.ti = ti

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
        ## slow down as we reach the parking spot
        ## TODO: Make this a little less hardcoded
        # dist = cur.distance(destination)
        # if dist < 3.0:
        #     print(f"[speed] SLOWING DOWN")
            
        #     target_speed = max(0.0, MIN_SPEED * 0.75)
        #     print(f"[slow] dist={dist:.2f} target={target_speed:.2f} cur.speed={cur.speed:.2f}")
        #     wp.speed = target_speed
        ##########
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

        has_stagnated = len(self.stagnation_history) == STAGNATION_HISTORY_LENGTH and np.linalg.norm(np.mean(self.stagnation_history, axis=0) - np.array([cur.x, cur.y])) < STAGNATION_THRESHOLD
        has_failed = len(self.failure_history) == FAILURE_HISTORY_LENGTH and np.linalg.norm(np.mean(self.failure_history, axis=0) - np.array([cur.x, cur.y])) < FAILURE_THRESHOLD

        trajectory = self.trajectory

        if has_failed:
            self.ti = 0
            self.trajectory = []
            self.mode = Mode.FAILED
            return

        new_trajectory = self._call_planner(cur, destination, self.obs)

        if new_trajectory:
            astar_pts = []
            if _map_processor._global_path is not None:
                for pt in _map_processor._global_path:
                    cx, cy, _ = standard_to_carla(pt[0], pt[1], 0.0)
                    astar_pts.append(TrajectoryPoint(Direction.FORWARD, cx, cy, MIN_SPEED, 0))
            self.debug_paths(new_trajectory, astar_pts)

            for i in range(1, TRAJECTORY_EXTENSION+1):
                new_trajectory.append(destination.offset(i/3))

            # min_dist = float('inf')
            # best_ti = 1
            # for i, tp in enumerate(new_trajectory):
            #     d = cur.distance(tp)
            #     if d < min_dist:
            #         min_dist = d
            #         best_ti = i
            
            # self.ti = max(1, best_ti)
            self.trajectory = new_trajectory  # only set here, not before

            self.ti = 1
            if has_stagnated:
                print('stagnated')
            self.stagnation_history = []
            self.mode = Mode.DRIVING
            self._last_collision_replan_time = self.time

            # print(f"best_ti: {best_ti}, trajectory len: {len(new_trajectory)}, total span: {new_trajectory[0].distance(new_trajectory[-1]):.2f}m")
        else:
            self.mode = Mode.STALLED

        if self.mode == Mode.STALLED and len(self.trajectory) > 0:
            self.mode = Mode.DRIVING

    def run_step(self, parked_car_ids=set()):
        self.perceive(parked_car_ids)
        self.plan()
        return self.control()

    def debug_paths(self, diffusion_trajectory, astar_trajectory=None):
        """
        Store trajectories for recording overlay.
        Green = diffusion, Yellow = A*.
        """
        self._overlay_diffusion = [(tp.x, tp.y, 0.3) for tp in diffusion_trajectory]
        self._overlay_astar = [(tp.x, tp.y, 0.3) for tp in (astar_trajectory or [])]

    def _compute_local_curvature(self, trajectory, ti, n=5):
        """Estimate curvature over next n waypoints"""
        pts = trajectory[ti:ti+n]
        if len(pts) < 3:
            return 0.0
        angles = []
        for i in range(1, len(pts)-1):
            v1 = np.array([pts[i].x - pts[i-1].x, pts[i].y - pts[i-1].y])
            v2 = np.array([pts[i+1].x - pts[i].x, pts[i+1].y - pts[i].y])
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angles.append(np.arccos(np.clip(cos_a, -1, 1)))
        return np.mean(angles)

    def _get_lookahead_dist(self, trajectory, ti):
        curvature = self._compute_local_curvature(trajectory, ti)
        # High curvature → short lookahead, low curvature → long lookahead
        if curvature > 0.3:    # sharp turn
            return 2.0
        elif curvature > 0.1:  # moderate turn
            return 4.0
        else:                  # straight
            return 8.0

