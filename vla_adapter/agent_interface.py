import sys
sys.path.insert(0, '/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo')
from team_code.agent_simlingo import LingoAgent
from team_code.nav_planner import RoutePlanner, _get_latlon_ref, _location_to_gps
from agents.navigation.local_planner import RoadOption

import carla
import numpy as np
import os
import pathlib
import cv2
from v2 import CarlaGnssSensor, CarlaTimeSensor, CarlaCollisionSensor
from team_code.transfuser_utils import inverse_conversion_2d, preprocess_compass

save_dir = "/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/results/3zvision"


class SimLingoAdapter(LingoAgent):
    def init_testbed(self, checkpoint_path, world, actor, destination, angle, save_path='/tmp/simlingo'):
        # Path to where visualizations and other debug output gets stored, from agent_simlingo.py
        os.environ['SAVE_PATH'] = save_path + '/'
        os.makedirs(save_path, exist_ok=True)
        os.environ['HUGGINGFACE_HUB_CACHE'] = '/tmp/hf_cache/hub'
        os.environ['TRANSFORMERS_CACHE'] = '/tmp/hf_cache/hub'
        os.environ['HF_HOME'] = '/tmp/hf_cache'
        # Remove offline mode - let it use cache naturally
        os.environ.pop('TRANSFORMERS_OFFLINE', None)
        
        # -------------------------------------------------------------

        pathlib.Path.__add__ = lambda self, other: str(self) + other

        ### HACK: Pathing
        self.setup(checkpoint_path, route_index = 'testbed')
        self.debug_save_path = self.save_path + '/debug_viz'
        pathlib.Path(self.debug_save_path).mkdir(parents=True, exist_ok=True)
        self.save_path_metric = self.debug_save_path + '/metric'
        pathlib.Path(self.save_path_metric).mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------------------

        self.initialized = True
        self.control = carla.VehicleControl(0.0, 0.0, 1.0)
        self.latest_pred_route = None

        ### HACK: bugfix of scenario override
        self.config.clip_throttle = 0.75
        self.config.clip_delta = 0.25
        self.config.brake_ratio = 1.01
        self.config.brake_speed = 0.05
        self.config.creep_throttle = 0.05    

        ### Prompting the car!!!
        dest_angle = angle
        self.custom_prompt = f"Park the vehicle in the parking spot ahead. The target parking spot is at location {destination.x}, {destination.y} in the CARLA frame. The target parking orientation is {np.rad2deg(dest_angle):.0f} degrees."
        self.metric_info = {}
        self.hero_actor = actor
        
        map = world.get_map()
        # This ensures GPS conversion is 1:1 with your current map
        lat_ref, lon_ref = _get_latlon_ref(map)
        self._lat_ref = lat_ref
        self._lon_ref = lon_ref

        # ### harcoded constants from Bench2Planner due to above not working
        # self._lat_ref = 42.0
        # self._lon_ref = 2.0
        self.route_planner_min_distance = 0.5

        self._route_planner = RoutePlanner(
            self.route_planner_min_distance,
            self.route_planner_max_distance,
            self._lat_ref, self._lon_ref
        )

        loc = actor.get_location()
        
        

        dx = np.cos(dest_angle) * 3.0
        dy = np.sin(dest_angle) * 3.0
        dest = carla.Location(x=destination.x, y=destination.y, z=loc.z)
        past_dest = carla.Location(x=destination.x + dx, y=destination.y + dy, z=loc.z)
        self._destination = destination
        self._dest_angle = dest_angle

        NUM_APPROACH_POINTS = 1
        SPACING = 0.5  # meters between points

        route = [(carla.Transform(loc), RoadOption.LANEFOLLOW)]  # start
        route.append((carla.Transform(dest), RoadOption.LANEFOLLOW))

        for i in range(1, NUM_APPROACH_POINTS + 1):
            point = carla.Location(
                x=destination.x + i * SPACING * np.cos(dest_angle),
                y=destination.y + i * SPACING * np.sin(dest_angle),
                z=loc.z
            )
            route.append((carla.Transform(point), RoadOption.LANEFOLLOW))

        self._route_planner.set_route(route, gps=False)

        # self._route_planner.set_route([
        #     (carla.Transform(loc), RoadOption.VOID),
        #     (carla.Transform(dest), RoadOption.VOID),
        #     (carla.Transform(past_dest), RoadOption.VOID),
        # ], gps=False)

        cfg = self.config

        ## building the camera
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(cfg.camera_width_0))
        cam_bp.set_attribute('image_size_y', str(cfg.camera_height_0))
        cam_bp.set_attribute('fov', str(cfg.camera_fov_0))

        self._cam = world.spawn_actor(
            cam_bp,
            carla.Transform(
                carla.Location(*cfg.camera_pos_0),
                carla.Rotation(
                    roll=cfg.camera_rot_0[0],
                    pitch = cfg.camera_rot_0[1],
                    yaw = cfg.camera_rot_0[2]
                )
            ),
            attach_to=actor,
            attachment_type=carla.AttachmentType.Rigid
        )
        self.latest_frame = None

        print(f"spawn: {loc.x:.2f}, {loc.y:.2f}")
        print(f"destination: {destination.x:.2f}, {destination.y:.2f}")
        print(f"distance: {np.sqrt((loc.x-destination.x)**2 + (loc.y-destination.y)**2):.2f}m")
        self._cam.listen(lambda img: setattr(self, 'latest_frame', img))

        

        print(f"camera_pos_0: {cfg.camera_pos_0}")
        print(f"camera_rot_0: {cfg.camera_rot_0}")
        print(f"camera_fov_0: {cfg.camera_fov_0}")
        print(f"camera_width_0: {cfg.camera_width_0}")
        print(f"camera_height_0: {cfg.camera_height_0}")
        self.draw_route_waypoints(world)

    def run_step_testbed(self, timestamp):
        if self.latest_frame is None:
            print("No frame yet")
            return carla.VehicleControl(brake=1.0)
        print("STEP:", self.step)
        # print(f"step={self.step}, control={self.control}")
        if self.latest_frame is None:
            return carla.VehicleControl(brake=1.0)

        actor = self._cam.parent
        loc = actor.get_location()


        

        transform = actor.get_transform()
        yaw = np.deg2rad(transform.rotation.yaw)
        pitch = np.deg2rad(transform.rotation.pitch)
        orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])

        
        vel = actor.get_velocity()
        vel_np = np.array([vel.x, vel.y, vel.z])
        speed = np.dot(vel_np, orientation)

        print(f"dest_angle: {np.rad2deg(self._dest_angle):.1f} deg")
        print(f"car yaw: {np.rad2deg(yaw):.1f} deg")
        print(f"direct vec to dest: dx={self._destination.x - loc.x:.2f}, dy={self._destination.y - loc.y:.2f}")
        forward = actor.get_transform().get_forward_vector()
        print(f"forward vector: x={forward.x:.2f}, y={forward.y:.2f}")
        print(f"yaw: {transform.rotation.yaw:.1f}")



        gps_dict = _location_to_gps(self._lat_ref, self._lon_ref, loc)
        gps = np.array([gps_dict['lat'], gps_dict['lon'], gps_dict['z']])

        data = np.frombuffer(self.latest_frame.raw_data, dtype=np.uint8)
        rgb = data.reshape((self.latest_frame.height, self.latest_frame.width, 4))[:, :, :3]

        # cv2.imwrite(f'{"/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/results/3zvision"}/frame_{self.step:04d}.jpg', rgb[:, :, ::-1])


        acc = actor.get_acceleration()
        ang_vel = actor.get_angular_velocity()

        input_data = {
            'rgb_0': (None, rgb),
            'speed': (None, {'speed': speed}),
            'gps':   (None, gps),
            'imu': (None, np.array([
                acc.x, acc.y, acc.z,
                ang_vel.x, ang_vel.y, ang_vel.z,
                yaw + np.deg2rad(90.0)
            ])),
        }


        print(f"gps: {gps}")
        print(f"loc: {loc.x}, {loc.y}")

        ## debug print
        gps_pos = self._route_planner.convert_gps_to_carla(gps)
        print(f"gps_pos (route planner coords): {gps_pos[:2]}")
        print(f"actual loc (world coords): {loc.x:.2f}, {loc.y:.2f}")
        print(f"route waypoint 0: {self._route_planner.route[0][0][:2]}")
        print(f"route waypoint 1: {self._route_planner.route[1][0][:2]}")
        waypoint_route = self._route_planner.run_step(np.append(gps_pos[:2], gps[2]))
        if len(waypoint_route) > 1:
            target_point = waypoint_route[1][0]
            
            compass = preprocess_compass(input_data['imu'][1][-1])
            ego_tp = inverse_conversion_2d(target_point[:2], gps_pos[:2], compass)

            self.draw_planned_path(target_point, compass, ego_tp, rgb)

            print(f"ego_target_point: {ego_tp}")
            print(f"target world coords: {target_point[:2]}")
            print(f"Destination: {self._destination}")


        return self.run_step(input_data, timestamp)

    # HACK
    def get_metric_info(self):
        actor = self._cam.parent
        def v2l(v, rot=False):
            return [v.roll, v.pitch, v.yaw] if rot else [v.x, v.y, v.z]
        return {
            'acceleration': v2l(actor.get_acceleration()),
            'angular_velocity': v2l(actor.get_angular_velocity()),
            'forward_vector': v2l(actor.get_transform().get_forward_vector()),
            'right_vector': v2l(actor.get_transform().get_right_vector()),
            'location': v2l(actor.get_transform().location),
            'rotation': v2l(actor.get_transform().rotation, rot=True),
        }

    # def control_pid(self, route_waypoints, velocity, speed_waypoints):
    #     steer, throttle, brake = super().control_pid(route_waypoints, velocity, speed_waypoints)
    #     throttle = min(throttle, 1)  # cap for parking speed
    #     return steer, throttle, brake

    

    def is_done(self, destination, threshold = 0.5):
        loc = self._cam.parent.get_location()
        dist = np.sqrt((loc.x - destination.x)**2 + (loc.y - destination.y)**2)
        return dist < threshold

    def destroy_cam(self):
        if hasattr(self, '_cam') and self._cam.is_alive:
            self._cam.destroy()


    

    def draw_route_waypoints(self, world, duration=10.0):
        return
        debug = world.debug
        route = self._route_planner.route

        for i, (wp, _) in enumerate(route):
            x, y, z = wp[0], wp[1], wp[2]

            color = carla.Color(r=255, g=0, b=0) if i == len(route) - 1 else carla.Color(r=0, g=255, b=0)
            debug.draw_point(
                carla.Location(x=x, y=y, z=z + 0.5),
                size=0.1,
                color=color,
                life_time=duration
            )

            if i < len(route) - 1:
                nx, ny, nz = route[i + 1][0]
                debug.draw_line(
                    carla.Location(x=x, y=y, z=z + 0.5),
                    carla.Location(x=nx, y=ny, z=nz + 0.5),
                    thickness=0.05,
                    color=carla.Color(r=0, g=200, b=255),
                    life_time=duration
                )

    def draw_planned_path(self, target_point, compass, ego_tp, rgb):
        
        # project ego_tp onto image
        # ego frame: x=forward, y=left
        # camera intrinsics from config
        cfg = self.config
        fx = cfg.camera_width_0 / (2.0 * np.tan(np.deg2rad(cfg.camera_fov_0 / 2)))
        cx = cfg.camera_width_0 / 2
        cy = cfg.camera_height_0 / 2
        
        # ego_tp is (forward, left) — convert to camera frame (x right, y down, z forward)
        cam_x = -ego_tp[1]  # left → right
        cam_y = -2.0        # camera height offset
        cam_z = ego_tp[0]   # forward

        rgb_save = rgb[:, :, ::-1].copy()
        
        if cam_z > 0:
            px = int(fx * cam_x / cam_z + cx)
            py = int(fx * cam_y / cam_z + cy)
            if 0 <= px < cfg.camera_width_0 and 0 <= py < cfg.camera_height_0:
                cv2.circle(rgb_save, (px, py), 5, (0, 165, 255), -1)  # orange dot


        
        cv2.imwrite(f'{save_dir}/frame_{self.step:04d}.jpg', rgb_save)