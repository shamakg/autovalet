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
import torch
from v2 import CarlaGnssSensor, CarlaTimeSensor, CarlaCollisionSensor, TrajectoryPoint, Direction, refine_trajectory
from team_code.transfuser_utils import inverse_conversion_2d, preprocess_compass
from nav_planner import LateralPIDController
save_dir = "/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/results/save_trajectory"

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
        self.config.clip_throttle = 0.4
        # self.config.clip_delta = 0.1
        self.config.brake_ratio = 1.001
        self.config.brake_speed = 0
        self._done_counter = 0
        self._done_threshold = 60
        # self.config.creep_throttle = 0.05
        
        # self.turn_controller = LateralPIDController(inference_mode=False) 
        # self.turn_controller.k_i = 0.1
        self.save_path_metric = None
        # self.turn_controller.default_lookahead = 75
        # self.default_lookahead = 5

        # ---------------------------------------------------------------   

        ### Prompting the car!!!
        dest_angle = angle
        self.custom_prompt = f"Park the vehicle in the parking spot ahead. The target parking spot is at location {destination.x}, {destination.y} in the CARLA frame. The target parking orientation is {np.rad2deg(dest_angle):.0f} degrees."
        self.metric_info = {}
        self.hero_actor = actor

        self._astar_ti = 0

        # ---------------------------------------------------------------   
        
        # Blueprint Debug
        pc = self.hero_actor.get_physics_control()
        print("mass:", pc.mass)                                                                                                                                   
        for i, w in enumerate(pc.wheels):
            print(f"wheels: {[(w.position.x, w.position.y) for w in pc.wheels]}")                                                                                                                         
            print(i, "max_steer:", w.max_steer_angle, "friction:", w.tire_friction, "pos:", w.position) 
        
        # ---------------------------------------------------------------   

        loc = actor.get_location()   
        map = world.get_map()
        # This ensures GPS conversion is 1:1 with your current map
        lat_ref, lon_ref = _get_latlon_ref(map)
        self._lat_ref = lat_ref
        self._lon_ref = lon_ref

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

        ### offset to get the INSIDE of the parking spot
        self._destination_bb = [                                                                                                     
            destination.x - 2.4, destination.y - 0.96,                                                                               
            destination.x + 2.4, destination.y + 0.96                                                                                
        ]
        self._dest_angle = dest_angle

        # ---------------------------------------------------------------

        NUM_APPROACH_POINTS = 1
        SPACING = 0.5  # meters between points

        route = [(carla.Transform(loc), RoadOption.LANEFOLLOW)]  # start
        
        ### Add start position, + two positions for straightening out into the spot
        point = carla.Location(
            x=destination.x - 2 * SPACING * np.cos(dest_angle),
            y=destination.y - 2 * SPACING * np.sin(dest_angle),
            z=loc.z
        )
        route.append((carla.Transform(point), RoadOption.LANEFOLLOW))
        route.append((carla.Transform(dest), RoadOption.LANEFOLLOW))
        point = carla.Location(
            x=destination.x + 1 * SPACING * np.cos(dest_angle),
            y=destination.y + 1 * SPACING * np.sin(dest_angle),
            z=loc.z
        )
        route.append((carla.Transform(point), RoadOption.LANEFOLLOW))
        self._route_planner.set_route(route, gps=False)

        # ---------------------------------------------------------------
    
        cfg = self.config

        # ---------------------------------------------------------------
    
        ## building the camera
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(cfg.camera_width_0))
        cam_bp.set_attribute('image_size_y', str(cfg.camera_height_0))
        cam_bp.set_attribute('fov', str(cfg.camera_fov_0))
        
        # ---------------------------------------------------------------
    
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

        # ---------------------------------------------------------------

        # DRAW DEBUG
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

        # ---------------------------------------------------------------

    def run_step_testbed(self, timestamp):
        if self.latest_frame is None:
            print("No frame yet")
            return carla.VehicleControl(brake=1.0)

        print("STEP:", self.step)
        # print(f"step={self.step}, control={self.control}")
        if self.latest_frame is None:
            return carla.VehicleControl(brake=1.0)

        actor = self.hero_actor
        loc = actor.get_location()

        transform = actor.get_transform()
        yaw = np.deg2rad(transform.rotation.yaw)
        pitch = np.deg2rad(transform.rotation.pitch)
        orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        
        vel = actor.get_velocity()
        vel_np = np.array([vel.x, vel.y, vel.z])
        speed = np.dot(vel_np, orientation)

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
        
    # ---------------------------------------------------------------   

        ## debug print
        print(f"dest_angle: {np.rad2deg(self._dest_angle):.1f} deg")
        print(f"car yaw: {np.rad2deg(yaw):.1f} deg")
        print(f"direct vec to dest: dx={self._destination.x - loc.x:.2f}, dy={self._destination.y - loc.y:.2f}")
        forward = actor.get_transform().get_forward_vector()
        print(f"forward vector: x={forward.x:.2f}, y={forward.y:.2f}")
        print(f"yaw: {transform.rotation.yaw:.1f}")
        print(f"gps: {gps}")
        print(f"loc: {loc.x}, {loc.y}")
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

    # ---------------------------------------------------------------   


        result = self.run_step(input_data, timestamp)
        # if self.latest_pred_route is not None:
        #     np.save(f'{save_dir}/pred_route_{self.step:04d}.npy', self.latest_pred_route)

        return result

    def run_step_testbed_astar(self, timestamp):
        """Use hardcoded A* trajectory instead of model."""
        if self.latest_frame is None:
            print("No frame yet")
            return carla.VehicleControl(brake=1.0)

        if not hasattr(self, '_astar_trajectory'):
            print("ERROR: _astar_trajectory not initialized. Call init_astar_trajectory() first.")
            return carla.VehicleControl(brake=1.0)

        self.step += 1
        actor = self.hero_actor
        loc = actor.get_location()

        transform = actor.get_transform()
        yaw = np.deg2rad(transform.rotation.yaw)
        pitch = np.deg2rad(transform.rotation.pitch)
        orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])

        vel = actor.get_velocity()
        vel_np = np.array([vel.x, vel.y, vel.z])
        speed = np.dot(vel_np, orientation)

        # Advance ti while the next waypoint is closer than the current one
        traj = self._astar_trajectory
        ti = self._astar_ti
        while ti + 1 < len(traj):
            dc = (traj[ti].x - loc.x) ** 2 + (traj[ti].y - loc.y) ** 2
            dn = (traj[ti + 1].x - loc.x) ** 2 + (traj[ti + 1].y - loc.y) ** 2
            if dn < dc:
                ti += 1
            else:
                break
        self._astar_ti = ti

        # Draw trajectory
        world = self.hero_actor.get_world()
        for i in range(len(traj) - 1):
            p1 = traj[i]
            p2 = traj[i + 1]
            world.debug.draw_line(
                carla.Location(x=p1.x, y=p1.y, z=0.5),
                carla.Location(x=p2.x, y=p2.y, z=0.5),
                thickness=0.1,
                color=carla.Color(0, 255, 255),  # cyan
                life_time=5
            )
        # Highlight current waypoint in green
        if ti < len(traj):
            p = traj[ti]
            world.debug.draw_point(
                carla.Location(x=p.x, y=p.y, z=0.8),
                size=0.2,
                color=carla.Color(0, 255, 0),  # green
                life_time=5
            )

        # Slice next N points ahead, pad with last if near the end
        N = 30
        chunk = traj[ti:ti + N]
        if len(chunk) < N:
            chunk = list(chunk) + [traj[-1]] * (N - len(chunk))

        # World -> ego (same transform used in the real pipeline)
        compass = preprocess_compass(yaw + np.deg2rad(90.0))
        cur_xy = np.array([loc.x, loc.y], dtype=np.float32)
        ego_pts = np.stack([
            inverse_conversion_2d(np.array([p.x, p.y], dtype=np.float32), cur_xy, compass)
            for p in chunk
        ]).astype(np.float32)
        pred_route = torch.from_numpy(ego_pts[np.newaxis])

        # Fake speed waypoints: straight forward, 0.25 s spacing
        DESIRED = 0.75
        dt = 0.25
        speed_wps = np.array(
            [[DESIRED * dt * (i + 1), 0.0] for i in range(N)], dtype=np.float32
        )
        pred_speed_wps = torch.from_numpy(speed_wps[np.newaxis])

        gt_velocity = torch.FloatTensor([[float(speed)]])

        steer, throttle, brake = self.control_pid(pred_route, gt_velocity, pred_speed_wps)

        if self.step < self.config.inital_frames_delay:
            control = carla.VehicleControl(0.0, 0.0, 1.0)
        else:
            control = carla.VehicleControl(
                steer=float(steer), throttle=float(throttle), brake=float(brake)
            )
        self.control = control

        print(
            f"[astar] step={self.step} ti={ti}/{len(traj)} v={float(speed):.2f} "
            f"steer={float(steer):.3f} thr={float(throttle):.3f} brk={int(bool(brake))} "
            f"ego0=({float(ego_pts[0,0]):.2f},{float(ego_pts[0,1]):.2f})"
        )
        return control

    def init_astar_trajectory(self, trajectory_points):
        """Initialize the A* trajectory. trajectory_points should be a list of TrajectoryPoint objects."""
        self._astar_trajectory = trajectory_points
        self._astar_ti = 0
        print(f"Initialized A* trajectory with {len(trajectory_points)} points")



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

    def control_pid_base(self, route_waypoints, velocity, speed_waypoints):
        # temporarily print desired_speed
        route_wps = route_waypoints[0].data.cpu().numpy()
        speed_wps = speed_waypoints[0].data.cpu().numpy()
        one_second = int(self.config.carla_fps // (self.config.wp_dilation * self.config.data_save_freq))
        half_second = one_second // 2
        desired_speed = np.linalg.norm(speed_wps[half_second - 2] - speed_wps[one_second - 2]) * 2.0
        print(f"desired_speed: {desired_speed:.3f} m/s")
        
        
        steer, throttle, brake = super().control_pid(route_waypoints, velocity, speed_waypoints)
        print("Route", route_wps.shape, route_wps[:3])
        
        return steer, throttle, brake

    def control_pid(self, route_waypoints, velocity, speed_waypoints):
        if hasattr(self, '_astar_trajectory') and self._astar_trajectory:
            loc = self.hero_actor.get_location()
            yaw = np.deg2rad(self.hero_actor.get_transform().rotation.yaw)
            traj = self._astar_trajectory

            # advance ti
            ti = getattr(self, '_astar_ti', 0)
            while ti + 1 < len(traj):
                dc = (traj[ti].x - loc.x)**2 + (traj[ti].y - loc.y)**2
                dn = (traj[ti+1].x - loc.x)**2 + (traj[ti+1].y - loc.y)**2
                if dn < dc:
                    ti += 1
                else:
                    break
            self._astar_ti = ti

            chunk = traj[ti:ti+20]
            while len(chunk) < 20:
                chunk = list(chunk) + [traj[-1]]

            ego_pts = []
            for wp in chunk:
                dx = wp.x - loc.x
                dy = wp.y - loc.y
                ex = dx * np.cos(-yaw) - dy * np.sin(-yaw)
                ey = dx * np.sin(-yaw) + dy * np.cos(-yaw)
                ego_pts.append([ex, ey])

            route_waypoints = torch.tensor(np.array(ego_pts), dtype=torch.float32).unsqueeze(0)

            TARGET_SPEED = 1.5
            dt = 0.25
            speed_wps = np.array([[TARGET_SPEED * dt * (i+1), 0.0] for i in range(20)], dtype=np.float32)
            speed_waypoints = torch.tensor(speed_wps, dtype=torch.float32).unsqueeze(0)

            print(f"[astar] ti={ti}/{len(traj)} ego0=({ego_pts[0][0]:.2f},{ego_pts[0][1]:.2f})")

        return super().control_pid(route_waypoints, velocity, speed_waypoints)

        

    # def is_done(self, destination, threshold = 0.25, angle_threshold_deg = 15):
    #     loc = self.hero_actor.get_location()
    #     dist = np.sqrt((loc.x - destination.x)**2 + (loc.y - destination.y)**2)
    #     yaw = np.deg2rad(self.hero_actor.get_transform().rotation.yaw)
    #     angle_diff = abs(np.degrees(np.arctan2(np.sin(yaw - self._dest_angle), np.cos(yaw - self._dest_angle))))
    #     print(f"DIST {dist:.2f}  ANGLE_DIFF {angle_diff:.1f}")
    #     return dist < threshold  # and angle_diff < angle_threshold_deg

    def is_done(self, destination=None, threshold=0.25, angle_threshold_deg=15):
        loc = self.hero_actor.get_location()
        bb = self._destination_bb
        inside = (bb[0] <= loc.x <= bb[2]) and (bb[1] <= loc.y <= bb[3])
        # Always use self._destination (raw parking location) so the distance
        # check is consistent with the bounding box, which is also built from it.
        dest = self._destination
        dist = np.sqrt((loc.x - dest.x)**2 + (loc.y - dest.y)**2)
        yaw = np.deg2rad(self.hero_actor.get_transform().rotation.yaw)
        angle_diff = abs(np.degrees(np.arctan2(np.sin(yaw - self._dest_angle), np.cos(yaw - self._dest_angle))))
        print(f"DIST {dist:.2f}  ANGLE_DIFF {angle_diff:.1f}  INSIDE {inside}")
        
        currently_done = inside and dist < threshold
        
        if currently_done:
            self._done_counter += 1
        else:
            self._done_counter = 0
        
        return self._done_counter >= self._done_threshold 

        
    def destroy_cam(self):
        if hasattr(self, '_cam') and self._cam.is_alive:
            self._cam.destroy()

# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------


### DEBUGS

    # ------------------------------------------------------------------
    # DEBUG: bypass the model and feed a static ground-truth trajectory
    # into control_pid so we can isolate the nav_planner PID behavior.
    # ------------------------------------------------------------------
    def init_debug_trajectory(self, shape='sine'):
        """Build a static world-frame ground-truth trajectory once.

        Uses v2.py's TrajectoryPoint / refine_trajectory so the format
        matches what plan_hybrid_a_star produces.
        """
        actor = self.hero_actor

        yaw0 = np.deg2rad(actor.get_transform().rotation.yaw)
        cos_y, sin_y = np.cos(yaw0), np.sin(yaw0)

        loc = actor.get_location()

        N, ds = 150, 0.1  # 15 m of path
        traj = []
        for i in range(N):
            s = i * ds
            if shape == 'sine':
                amp, wl = 1.5, 6.0
                lon = s
                lat = amp * np.sin(2 * np.pi * s / wl)
                dlat = amp * (2 * np.pi / wl) * np.cos(2 * np.pi * s / wl)
                local_yaw = np.arctan2(dlat, 1.0)
                wx = loc.x + lon * cos_y - lat * sin_y
                wy = loc.y + lon * sin_y + lat * cos_y
                traj.append(TrajectoryPoint(Direction.FORWARD, wx, wy, 2.0, yaw0 + local_yaw))
            elif shape == 'arc':
                R = 6.0
                
                # straight section: 5m forward
                for i in range(50):
                    s = i * 0.1
                    lon = s
                    lat = 0.0
                    wx = loc.x + lon * cos_y - lat * sin_y
                    wy = loc.y + lon * sin_y + lat * cos_y
                    traj.append(TrajectoryPoint(Direction.FORWARD, wx, wy, 2.0, yaw0))
                
                # right turn: arc in lon/lat space
                # a right turn means lat goes negative (right = -lat in this convention)
                for i in range(1, 50):
                    theta = i * (np.pi/2) / 49  # 0 to pi/2
                    lon = 5.0 + R * np.sin(theta)
                    lat = -R * (1 - np.cos(theta))  # negative = right
                    local_yaw = -theta  # turning right = negative yaw change
                    wx = loc.x + lon * cos_y - lat * sin_y
                    wy = loc.y + lon * sin_y + lat * cos_y
                    traj.append(TrajectoryPoint(Direction.FORWARD, wx, wy, 1.0, yaw0 + local_yaw))
        refine_trajectory(traj)
        self._debug_trajectory = traj
        self._debug_ti = 0

        world = actor.get_world()
        for i in range(len(traj) - 1):
            a, b = traj[i], traj[i + 1]
            world.debug.draw_line(
                carla.Location(x=a.x, y=a.y, z=loc.z + 0.5),
                carla.Location(x=b.x, y=b.y, z=loc.z + 0.5),
                thickness=0.05,
                color=carla.Color(r=255, g=0, b=255),
                life_time=0.0,
            )
        print(f"[debug] ground-truth trajectory built: shape={shape}, N={len(traj)}")
            


    def run_step_testbed_debug(self, timestamp):
        """PID-only debug step: follow the static trajectory, skip the model."""
        if self.latest_frame is None:
            return carla.VehicleControl(brake=1.0)
        if not hasattr(self, '_debug_trajectory'):
            self.init_debug_trajectory('sine')

        self.step += 1

        actor = self._cam.parent
        loc = actor.get_location()
        yaw = np.deg2rad(actor.get_transform().rotation.yaw)
        vel = actor.get_velocity()
        fwd = actor.get_transform().get_forward_vector()
        speed_mps = vel.x * fwd.x + vel.y * fwd.y + vel.z * fwd.z

        # advance ti while the next waypoint is closer than the current one
        traj = self._debug_trajectory
        ti = self._debug_ti
        while ti + 1 < len(traj):
            dc = (traj[ti].x - loc.x) ** 2 + (traj[ti].y - loc.y) ** 2
            dn = (traj[ti + 1].x - loc.x) ** 2 + (traj[ti + 1].y - loc.y) ** 2
            if dn < dc:
                ti += 1
            else:
                break
        self._debug_ti = ti

        # slice next N points ahead, pad with last if near the end
        N = 30
        chunk = traj[ti:ti + N]
        if len(chunk) < N:
            chunk = list(chunk) + [traj[-1]] * (N - len(chunk))

        # world -> ego (same transform used in the real pipeline)
        compass = preprocess_compass(yaw + np.deg2rad(90.0))
        cur_xy = np.array([loc.x, loc.y], dtype=np.float32)
        ego_pts = np.stack([
            inverse_conversion_2d(np.array([p.x, p.y], dtype=np.float32), cur_xy, compass)
            for p in chunk
        ]).astype(np.float32)
        pred_route = torch.from_numpy(ego_pts[np.newaxis])

        # fake speed waypoints: straight forward, 0.25 s spacing
        # desired_speed = norm(wp[0] - wp[2]) * 2 = DESIRED (see control_pid)
        DESIRED = 0.75
        dt = 0.25
        speed_wps = np.array(
            [[DESIRED * dt * (i + 1), 0.0] for i in range(N)], dtype=np.float32
        )
        pred_speed_wps = torch.from_numpy(speed_wps[np.newaxis])

        gt_velocity = torch.FloatTensor([[float(speed_mps)]])

        steer, throttle, brake = self.control_pid(pred_route, gt_velocity, pred_speed_wps)

        if self.step < self.config.inital_frames_delay:
            control = carla.VehicleControl(0.0, 0.0, 1.0)
        else:
            control = carla.VehicleControl(
                steer=float(steer), throttle=float(throttle), brake=float(brake)
            )
        self.control = control

        print(
            f"[debug] step={self.step} ti={ti}/{len(traj)} v={float(speed_mps):.2f} "
            f"steer={float(steer):.3f} thr={float(throttle):.3f} brk={int(bool(brake))} "
            f"ego0=({float(ego_pts[0,0]):.2f},{float(ego_pts[0,1]):.2f})"
        )
        return control

    def run_step_testbed_replay(self, timestamp):
        self.step += 1
        path = f'{save_dir}/pred_route_{self.step:04d}.npy'
        if not os.path.exists(path):
            print(f"No more saved routes at step {self.step}, stopping")
            return carla.VehicleControl(brake=1.0)
        
        route = np.load(f'{save_dir}/pred_route_{self.step:04d}.npy')
        pred_route = torch.from_numpy(route[np.newaxis]).float()
        
        # use real vehicle state
        actor = self.hero_actor
        vel = actor.get_velocity()
        fwd = actor.get_transform().get_forward_vector()
        speed_mps = vel.x * fwd.x + vel.y * fwd.y
        gt_velocity = torch.FloatTensor([[speed_mps]])
        
        # dummy speed waypoints
        speed_wps = np.zeros((20, 2), dtype=np.float32)
        pred_speed_wps = torch.from_numpy(speed_wps[np.newaxis])
        
        steer, throttle, brake = self.control_pid(pred_route, gt_velocity, pred_speed_wps)
        self.control = carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake))
        return self.control

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
        cam_y = 2.0        # camera height offset
        cam_z = ego_tp[0]   # forward

        rgb_save = rgb[:, :, ::-1].copy()
        
        if cam_z > 0:
            px = int(fx * cam_x / cam_z + cx)
            py = int(fx * cam_y / cam_z + cy)
            if 0 <= px < cfg.camera_width_0 and 0 <= py < cfg.camera_height_0:
                cv2.circle(rgb_save, (px, py), 5, (0, 165, 255), -1)  # orange dot


        
        # cv2.imwrite(f'{save_dir}/frame_{self.step:04d}.jpg', rgb_save)

        def pure_pursuit(self, route_np, speed_ms, wheelbase = 2.856):  # using Lincoln wheelbase
            L = max(1.0, 0.5 * speed_ms)
            dists = np.linalg.norm(route_np, axis=1)
            beyond = np.where(dists >= L)[0]

            target = route_np[beyond[0]] if len(beyond) > 0 else route_np[-1]
            alpha = np.arctan2(target[1], target[0])

            steer = np.arctan2(2 * wheelbase * np.sin(alpha), L)
            steer = np.clip(steer / np.deg2rad(70), -1.0, 1.0)

            return float(steer)
        
        def control_pid_pp(self, route_waypoints, velocity, speed_waypoints):
            route_wps = route_waypoints[0].data.cpu().numpy()
            speed_ms = float(velocity[0].data.cpu().numpy())

            steer = self.pure_pursuit(route_wps, speed_ms)
            steer = round(np.clip(steer, -1.0, 1.0), 3)

            _, throttle, brake = super().control_pid(route_waypoints, velocity, speed_waypoints)
            # throttle = min(throttle, 0.3)

            max_lateral = np.max(np.abs(route_wps[:, 1]))  # check ALL waypoints not just first 10
            MAX_SPEED = 1.5  # hard cap
            if speed_ms > MAX_SPEED:
                throttle = 0.0
                brake = True
            elif max_lateral > 2.0 and speed_ms > 1.0:  # significant turn coming
                throttle = 0.0
                brake = True

            print(f"steer: {steer:.3f}, speed: {speed_ms:.2f}, max_lat: {max_lateral:.2f}")
            return steer, throttle, brake