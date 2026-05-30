from transfuser_utils import inverse_conversion_2d, preprocess_compass
import gzip
import ujson
import json
import os
import queue
import pathlib

import carla
from PIL import Image

import numpy as np
import random


import shutil

import sys, os
_AUTOVALET = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _AUTOVALET not in sys.path:
    sys.path.insert(0, _AUTOVALET)
from v2_experiment import TRAIN_SCENARIOS
from default_runner import ScenarioMode, run_scenario
from parking_position import parking_vehicle_locations_Town04 as PARKING_SPOTS

SAVE_EVERY_N = 5

_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__),
    '../simlingo/data/augmented_templates/lmdrive.json')
with open(_TEMPLATE_PATH) as _f:
    command_templates = json.load(_f)

### Templates for recovery data. Somewhat break the prompting logic, but not a huge problem for now
_REVERSE_TEMPLATES = {
    'left': [
        "The parking space is [x] meters behind you and [y] meters to your left.",
        "Park in the spot [x] meters behind and [y] meters to your left.",
        "Your parking target is [x] meters back and [y] meters to your left.",
        "The target parking space is [x] m behind you and [y] m to your left.",
    ],
    'right': [
        "The parking space is [x] meters behind you and [y] meters to your right.",
        "Park in the spot [x] meters behind and [y] meters to your right.",
        "Your parking target is [x] meters back and [y] meters to your right.",
        "The target parking space is [x] m behind you and [y] m to your right.",
    ],
}


def collect_dataset(world, output_dir):
    ### TOTAL: 500 episodes (the last successful data collection hadd 444/500 succcesses)
    episode_types = (
        ['normal']               * 50  +
        ['normal_empty']         * 50  +
        ['normal_close']         * 50  +
        ['normal_close_empty']   * 50  +
        ['recovery']             * 50  +
        ['pedestrian_normal']    * 150 +
        ['pedestrian_recovery']  * 50  +
        ['door_normal']          * 50
    )
    random.shuffle(episode_types)

    output_dir = pathlib.Path(output_dir)
    succeeded, failed = 0, 0

    ## just some pathing logic
    _episode_base = output_dir / 'data' / 'simlingo' / 'parking_ft' / 'routes_training' / 'RouteScenario_parking'
    if _episode_base.exists():
        shutil.rmtree(_episode_base)
    _episode_base.mkdir(parents=True)
    ## Loop through and save episodes
    for i, episode_type in enumerate(episode_types):
        destination, parked_spots = random.choice(TRAIN_SCENARIOS)
        save_path = _episode_base / f'Town04_{i:04d}'
        print(f"[{i+1}/{len(episode_types)}] {episode_type} → {save_path}")
        try:
            ### We run collect episode (which implements run_scenario from the original v2_experiment for consistency)
            ok = collect_episode(world, save_path, episode_type, destination, parked_spots)
        except Exception as e:
            print(f"  Episode CRASHED: {e}")
            import traceback; traceback.print_exc()
            if save_path.exists():
                shutil.rmtree(save_path)
            ok = False
        if ok:
            succeeded += 1
        else:
            failed += 1

    print(f"\nDone: {succeeded} succeeded, {failed} failed out of {len(episode_types)} episodes")

# not sure if this needs to be seeded, going with no seeding for now
## Starting point randomization (hopefully will improve robustness)
## In the future, would definitely like to try DAgger
def get_offsets(episode_type):
    if episode_type == 'normal':
        # Lane entrance only, mild variation
        y_offset = 0
        x_offset = np.random.uniform(-1.5, 1.5)

    elif episode_type == 'recovery':
        # Throughout lane, worse starting poses
        y_offset = np.random.uniform(10, 55)
        x_offset = np.random.uniform(-2.5, 2.5)

    elif episode_type == 'pedestrian_normal':
        # Fixed entrance — don't mess with pedestrian timing
        y_offset = 0
        x_offset = np.random.uniform(-1.0, 1.0)

    elif episode_type == 'pedestrian_recovery':
        # Slightly further in lane with pedestrians present
        # Keep x tight — interaction geometry is sensitive
        y_offset = np.random.uniform(10, 50)
        x_offset = np.random.uniform(-1.0, 1.0)

    elif episode_type == 'door_normal':
        # Fixed entrance
        y_offset = 0
        x_offset = np.random.uniform(-1.0, 1.0)
    
    elif episode_type in ('normal_close', 'normal_close_empty'):
        # Start close to spot entrance
        y_offset = np.random.uniform(35, 45)  # closer than normal
        x_offset = np.random.uniform(-1.5, 1.5)

    elif episode_type == 'normal_empty':
        y_offset = 0
        x_offset = np.random.uniform(-1.5, 1.5)

    return y_offset, x_offset

# taken from config_simlingo.py
## Important this is EXACTLY like how it is in config_simlingo.py
#  # -----------------------------------------------------------------------------
#         # Sensor config
#         # -----------------------------------------------------------------------------
#         self.num_cameras = [0] #,3] #,1,2]
#         # cam 1
#         self.camera_pos_0 = [-1.5, 0.0, 2.0]  # x, y, z mounting position of the camera
#         self.camera_rot_0 = [0.0, 0.0, 0.0]  # Roll Pitch Yaw of camera 0 in degree

#         self.camera_width_0 = 1024  # Camera width in pixel during data collection
#         self.camera_height_0 = 512  # Camera height in pixel during data collection
#         self.camera_fov_0 = 110

def setup_camera(world, vehicle, y_offset=0.0, yaw_offset=0.0):
    bp = world.get_blueprint_library().find('sensor.camera.rgb')
    bp.set_attribute('image_size_x', '1024')
    bp.set_attribute('image_size_y', '512')
    bp.set_attribute('fov', '110')

    transform = carla.Transform(
        carla.Location(x=-1.5, y=y_offset, z=2.0),
        carla.Rotation(roll=0.0, pitch=0.0, yaw=yaw_offset)
    )
    camera = world.spawn_actor(bp, transform, attach_to=vehicle)
    return camera

def save_measurement(save_path, frame, scenario, spot_idx, aug_translation=0.0, aug_rotation=0.0):
    actor = scenario.car.actor
    transform = actor.get_transform()
    traj = scenario.car.car.trajectory

    ### Ego matrix to go into SimLingo
    ego_matrix = np.asarray(transform.get_matrix()).tolist()
    ego_xy = np.array([transform.location.x, transform.location.y])
    yaw = np.deg2rad(transform.rotation.yaw)
    pitch = np.deg2rad(transform.rotation.pitch)
    vel = actor.get_velocity()
    # Signed speed: negative when reversing, matching inference-time convention.
    orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    speed = float(np.dot(np.array([vel.x, vel.y, vel.z]), orientation))

    # Process the actual route.
    # car.car.ti tracks the rear-axle reference point (~1.6 m behind the vehicle
    # center), so using it directly makes route[0] land ~1.6 m behind the car in
    # ego frame. Instead, find the trajectory waypoint closest to the vehicle center.
    spot = PARKING_SPOTS[spot_idx]
    target_pt = inverse_conversion_2d(np.array([spot.x, spot.y]), ego_xy, yaw).tolist()
    # The route/path always points toward the destination, regardless of whether
    # the ego is yielding to a dynamic obstacle. When the ego is stopped, the
    # speed waypoints (derived separately from future ego poses) cluster at the
    # origin on their own, so the "hold position" signal is preserved without
    # corrupting the path supervision.
    if len(traj) > 0:
        traj_xy = np.array([[wp.x, wp.y] for wp in traj])
        dists = np.linalg.norm(traj_xy - ego_xy, axis=1)
        ti = int(np.argmin(dists))
    else:
        ti = 0
    forward_traj = traj[ti:] if ti < len(traj) else traj
    if len(forward_traj) > 1:
        # route[0] is always the car's center ([0,0] in ego frame).
        # Sample 19 from forward_traj[1:] (strictly after the closest point)
        # so route[1] never ends up behind route[0].
        rest_traj = forward_traj[1:]
        indices = np.linspace(0, len(rest_traj) - 1, 19, dtype=int)
        rest = [inverse_conversion_2d(np.array([rest_traj[i].x, rest_traj[i].y]), ego_xy, yaw).tolist() for i in indices]
        route = [[0.0, 0.0]] + rest
    else:
        # Fallback: straight line from ego to destination
        route = np.linspace([0.0, 0.0], target_pt, 20).tolist()

    target_x = round(abs(target_pt[0]), 1)
    target_y = round(abs(target_pt[1]), 1)
    is_left   = target_pt[1] > 0
    is_behind = target_pt[0] < 0
    if is_behind:
        candidates = _REVERSE_TEMPLATES['left' if is_left else 'right']
    else:
        candidates = [t for t in command_templates["65"]
                      if not ("left"  in t and not is_left)
                      and not ("right" in t and     is_left)]
    command_str = random.choice(candidates).replace("[x]", str(target_x)).replace("[y]", str(target_y))

    ### data from simlingo (data load)
    data = {
          "ego_matrix":               ego_matrix,
          "speed":                    speed,
          "target_point":             target_pt,
          "target_point_next":        target_pt,
          "route":                    route,
          "route_original":           route,
          "command":                  65,
          "next_command":             65,
          "lmdrive_command":          command_str,
          "augmentation_translation": aug_translation,
          "augmentation_rotation":    aug_rotation,
      }

    out = save_path / "measurements" / f"{frame:04d}.json.gz"
    with gzip.open(out, "wt", encoding="utf-8") as f:
        ujson.dump(data, f, indent=4)


def save_boxes(save_path, frame, scenario):
    actor   = scenario.car.actor
    tf      = actor.get_transform()
    ego_yaw = np.deg2rad(tf.rotation.yaw)
    ego_xy  = np.array([tf.location.x, tf.location.y])

    boxes = []

    for parked in scenario.parked_cars:
        if not parked.is_alive:
            continue
        pt  = parked.get_transform()
        ext = parked.bounding_box.extent
        rel = inverse_conversion_2d(np.array([pt.location.x, pt.location.y]), ego_xy, ego_yaw)
        boxes.append({
            'class':    'car',
            'extent':   [ext.x, ext.y, ext.z],
            'position': [float(rel[0]), float(rel[1]), float(pt.location.z)],
            'yaw':      float(np.deg2rad(pt.rotation.yaw) - ego_yaw),
            'speed':    0.0,
            'brake':    1.0,
        })

    for sub in getattr(scenario, 'list_scenarios', []):
        for walker in getattr(sub, 'other_actors', []):
            if not walker.is_alive or 'walker' not in walker.type_id:
                continue
            wt  = walker.get_transform()
            ext = walker.bounding_box.extent
            vel = walker.get_velocity()
            rel = inverse_conversion_2d(np.array([wt.location.x, wt.location.y]), ego_xy, ego_yaw)
            boxes.append({
                'class':    'walker',
                'extent':   [ext.x, ext.y, ext.z],
                'position': [float(rel[0]), float(rel[1]), float(wt.location.z)],
                'yaw':      float(np.deg2rad(wt.rotation.yaw) - ego_yaw),
                'speed':    float(np.sqrt(vel.x**2 + vel.y**2)),
            })

    out = save_path / "boxes" / f"{frame:04d}.json.gz"
    with gzip.open(out, "wt", encoding="utf-8") as f:
        ujson.dump(boxes, f)


_EPISODE_MODE_MAP = {
    'normal':               ScenarioMode.CONE,
    'normal_empty':         ScenarioMode.EMPTY,
    'normal_close':         ScenarioMode.CONE,
    'normal_close_empty':   ScenarioMode.EMPTY,
    'recovery':             ScenarioMode.CONE,
    'pedestrian_normal':    ScenarioMode.PEDESTRIAN,
    'pedestrian_recovery':  ScenarioMode.PEDESTRIAN,
    'door_normal':          ScenarioMode.DOORMODE,
}

def collect_episode(world, save_path, episode_type, destination, parked_spots):
    """
    Run one parking episode and save rgb frames + measurements for SimLingo finetuning.

    Returns True if the car successfully parked, False otherwise.
    """
    save_path = pathlib.Path(save_path)
    if save_path.exists():
        shutil.rmtree(save_path)
    (save_path / 'rgb').mkdir(parents=True)
    (save_path / 'rgb_augmented').mkdir(parents=True)
    (save_path / 'measurements').mkdir(parents=True)
    (save_path / 'boxes').mkdir(parents=True)

    y_offset, x_offset = get_offsets(episode_type)
    scenario_mode = _EPISODE_MODE_MAP[episode_type]

    # Per-episode augmentation: mirrors simlingo/team_code/config.py exactly.
    #   camera_translation_augmentation_min/max = -1.5 / 1.5  (metres)
    #   camera_rotation_augmentation_min/max    = -20.0 / 20.0 (degrees)
    # Stored in DEGREES in measurements — dataset_base.py calls np.deg2rad() on load.
    aug_translation  = float(np.random.uniform(-1.5, 1.5))   # metres lateral
    aug_rotation_deg = float(np.random.uniform(-20.0, 20.0)) # degrees yaw (stored as-is)

    frame_q     = queue.Queue(maxsize=2)
    aug_frame_q = queue.Queue(maxsize=2)
    camera      = [None]
    aug_camera  = [None]
    frame_idx   = [0]
    tick_count  = [0]

    def on_scenario_ready(scenario):
        camera[0] = setup_camera(world, scenario.car.actor)
        camera[0].listen(lambda img: frame_q.put(img) if not frame_q.full() else None)
        aug_camera[0] = setup_camera(
            world, scenario.car.actor,
            y_offset=aug_translation, yaw_offset=aug_rotation_deg,  # degrees, as CARLA expects
        )
        aug_camera[0].listen(lambda img: aug_frame_q.put(img) if not aug_frame_q.full() else None)

    def on_step(scenario):
        tick_count[0] += 1
        if tick_count[0] % SAVE_EVERY_N != 0:
            return
        if frame_q.empty():
            return
        img = frame_q.get()
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
        Image.fromarray(arr[:, :, :3][..., ::-1]).save(
            save_path / 'rgb' / f'{frame_idx[0]:04d}.jpg', quality=95)
        if not aug_frame_q.empty():
            aug_img = aug_frame_q.get()
            aug_arr = np.frombuffer(aug_img.raw_data, dtype=np.uint8).reshape((aug_img.height, aug_img.width, 4))
            Image.fromarray(aug_arr[:, :, :3][..., ::-1]).save(
                save_path / 'rgb_augmented' / f'{frame_idx[0]:04d}.jpg', quality=95)
        save_measurement(save_path, frame_idx[0], scenario, destination,
                         aug_translation=aug_translation, aug_rotation=aug_rotation_deg)
        save_boxes(save_path, frame_idx[0], scenario)
        frame_idx[0] += 1

    ious, weighted_ious = [], []
    collisions_ref, near_misses_ref, walker_collisions_ref, actual_collisions = [0], [0], [0], []

    try:
        run_scenario(
            world, destination, parked_spots,
            ious, weighted_ious, collisions_ref, near_misses_ref,
            actual_collisions, walker_collisions_ref,
            recording_path=None, car_list=[None],
            scenario_mode=scenario_mode,
            start_y_offset=y_offset, start_x_offset=x_offset,
            on_step=on_step, on_scenario_ready=on_scenario_ready,
        )
    finally:
        for cam in (camera[0], aug_camera[0]):
            if cam is not None:
                cam.stop()
                cam.destroy()

    success = (
        len(ious) > 0 and
        ious[0] > 0.7 and
        collisions_ref[0] == 0 and
        walker_collisions_ref[0] == 0
    )

    if success:
        results = {
            'status': 'completed',
            'scores': {'score_composed': 100.0, 'score_route': 100.0},
            'num_infractions': 0,
            'infractions': {'min_speed_infractions': [], 'outside_route_lanes': []},
        }
        with gzip.open(save_path / 'results.json.gz', 'wt', encoding='utf-8') as f:
            ujson.dump(results, f)

        ### SAVED METADATA FOR DEBUGGING
        meta = {
            'episode_type':       episode_type,
            'destination':        destination,
            'y_offset':           y_offset,
            'x_offset':           x_offset,
            'iou':                ious[0] if ious else None,
            'vehicle_collisions': collisions_ref[0],
            'walker_collisions':  walker_collisions_ref[0],
            'frames_saved':       frame_idx[0],
        }
        with open(save_path / 'episode_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
    else:
        shutil.rmtree(save_path)

    iou_str = f"{ious[0]:.2f}" if ious else "n/a"
    print(f"  Episode {'SUCCESS' if success else 'FAILED'} — {frame_idx[0]} frames, iou={iou_str}")
    return success


if __name__ == '__main__':
    from testbed.v2_experiment_utils import load_client, town04_load, town04_spectator_bev
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

    OUTPUT = "/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/finetune/run_001/"

    client = load_client()
    world = town04_load(client)
    CarlaDataProvider.set_client(client)
    CarlaDataProvider.set_world(world)
    town04_spectator_bev(world)

    # Match SimLingo's 20 Hz physics (PID controller also uses dt=0.05)
    settings = world.get_settings()
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    collect_dataset(world, OUTPUT)