import sys
sys.path.insert(0, '/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo')
from team_code.agent_simlingo import LingoAgent
from team_code.nav_planner import RoutePlanner, _get_latlon_ref, _location_to_gps
from agents.navigation.local_planner import RoadOption

import carla
import numpy as np
import os
import pathlib

class SimLingoAdapter(LingoAgent):
    def init_testbed(self, checkpoint_path, world, actor, destination, save_path='/tmp/simlingo'):
        # Path to where visualizations and other debug output gets stored, from agent_simlingo.py
        os.environ['SAVE_PATH'] = save_path + '/'
        os.makedirs(save_path, exist_ok=True)
        os.environ['HUGGINGFACE_HUB_CACHE'] = '/tmp/hf_cache/hub'
        os.environ['TRANSFORMERS_CACHE'] = '/tmp/hf_cache/hub'
        os.environ['HF_HOME'] = '/tmp/hf_cache'
        # Remove offline mode - let it use cache naturally
        os.environ.pop('TRANSFORMERS_OFFLINE', None)


        pathlib.Path.__add__ = lambda self, other: str(self) + other
        ### HACK: Pathing
        self.setup(checkpoint_path, route_index = 'testbed')
        self.debug_save_path = self.save_path + '/debug_viz'
        pathlib.Path(self.debug_save_path).mkdir(parents=True, exist_ok=True)
        self.save_path_metric = self.debug_save_path + '/metric'
        pathlib.Path(self.save_path_metric).mkdir(parents=True, exist_ok=True)

        self.initialized = True
        self.control = carla.VehicleControl(0.0, 0.0, 1.0)

        self.metric_info = {}
        
        ### Not sure if this is accurate
        lat_ref, lon_ref = _get_latlon_ref(world.get_map())
        print(f"LAT {lat_ref}, LON {lon_ref}")
        self._lat_ref = lat_ref
        self._lon_ref = lon_ref

        self._route_planner = RoutePlanner(
            self.route_planner_min_distance,
            self.route_planner_max_distance,
            lat_ref, lon_ref
        )

        loc = actor.get_location()
        self._route_planner.set_route([
            #### void since no actual route
            (carla.Transform(loc), RoadOption.VOID),
            (carla.Transform(carla.Location(
                x=destination.x, y= destination.y, z=loc.z
            )), RoadOption.VOID),
        ], gps=False)

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
        self._cam.listen(lambda img: setattr(self, 'latest_frame', img))


    def run_step_testbed(self, timestamp):
        if self.latest_frame is None:
            return carla.VehicleControl(brake=1.0)

        actor = self._cam.parent
        loc = actor.get_location()
        yaw = np.deg2rad(actor.get_transform().rotation.yaw)
        speed = actor.get_velocity().length()

        gps_dict = _location_to_gps(self._lat_ref, self._lon_ref, loc)
        gps = np.array([gps_dict['lat'], gps_dict['lon'], gps_dict['z']])

        data = np.frombuffer(self.latest_frame.raw_data, dtype=np.uint8)
        rgb = data.reshape((self.latest_frame.height, self.latest_frame.width, 4))[:, :, :3]

        input_data = {
            'rgb_0': (None, rgb),
            'speed': (None, {'speed': speed}),
            'gps':   (None, gps),
            'imu':   (None, np.array([0, 0, 0, 0, 0, 0, yaw + np.deg2rad(90.0)])),
        }

        return self.run_step(input_data, timestamp)

    def is_done(self, destination, threshold = 0.5):
        loc = self._cam.parent.get_location()
        dist = np.sqrt((loc.x - destination.x)**2 + (loc.y - destination.y)**2)
        return dist < threshold

    def destroy_cam(self):
        if hasattr(self, '_cam') and self._cam.is_alive:
            self._cam.destroy()


