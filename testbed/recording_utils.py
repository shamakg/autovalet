import cv2
import os
from math import sqrt
from enum import Enum
from typing import Tuple, List
from queue import Queue
import carla 
import signal
import sys  
import numpy as np

from v2 import Mode
import subprocess


def init_recording(car, recording_path, width=640, height=360, fps=20, top_down=False):
        world = car.world
        actor = car.actor
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(width))
        cam_bp.set_attribute('image_size_y', str(height))
        cam_bp.set_attribute('fov', str(90))
        if top_down:
            # With pitch=-90, UE4 applies yaw then pitch — image-up = (cosθ, sinθ).
            # Setting camera_yaw = car_yaw makes car forward = image up.
            loc = actor.get_location()
            initial_yaw_deg = actor.get_transform().rotation.yaw
            camera_yaw_deg  = initial_yaw_deg
            car.topdown_camera_yaw_deg = camera_yaw_deg
            car.topdown_camera_yaw_rad = np.deg2rad(camera_yaw_deg)
            # The camera is static at the spawn location; overlays must project
            # relative to this fixed point (= image center), not the moving car.
            car.topdown_camera_loc = loc
            cam_transform = carla.Transform(
                carla.Location(x=loc.x, y=loc.y, z=loc.z + 30),
                carla.Rotation(pitch=-90, yaw=camera_yaw_deg),
            )
            frames_attr = 'frames_topdown'
            writer_attr = 'recording_writer_topdown'
            path_attr   = 'recording_path_topdown'
            cam = world.spawn_actor(cam_bp, cam_transform)
        else:
            cam_transform = carla.Transform(
                carla.Location(x=-10, z=5),
                carla.Rotation(pitch=-20)
            )
            frames_attr = 'frames'
            writer_attr = 'recording_writer'
            path_attr   = 'recording_path'
            cam = world.spawn_actor(cam_bp, cam_transform, attach_to=actor, attachment_type=carla.AttachmentType.Rigid)
        writer = cv2.VideoWriter(
            recording_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        if not hasattr(car, frames_attr):
            setattr(car, frames_attr, Queue())
        
        car.recording_width = width
        car.recording_height = height
        setattr(car, writer_attr, writer)
        setattr(car, path_attr, recording_path)
        queue = getattr(car, frames_attr)
        
        if top_down:
            car.latest_topdown_frame = None
            cam.listen(lambda image: (queue.put(image), setattr(car, 'latest_topdown_frame', image)) and None)
        else:
            cam.listen(lambda image: queue.put(image))
        return cam

def process_recording_frames(car):
    for frames_attr, writer_attr in [
        ('frames',          'recording_writer'),
        ('frames_topdown',  'recording_writer_topdown'),
    ]:
        if not hasattr(car, frames_attr):
            continue
        frames = getattr(car, frames_attr)
        writer = getattr(car, writer_attr, None)
        while not frames.empty() and writer:
            if car.has_recorded_segment and car.car.mode == Mode.PARKED:
                return
            image = frames.get()
            car.has_recorded_segment = False
            data = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
            data = data[:, :, :3].copy()  # RGBA -> BGR (CARLA gives BGRA)
            # Overlay model's predicted trajectory on topdown frames
            if frames_attr == 'frames_topdown':
                # Project relative to the camera's fixed spawn location (image center),
                # not the moving car — otherwise overlays drift as the car moves.
                actor_loc = car.actor.get_location()
                camera_yaw_rad = getattr(car, 'topdown_camera_yaw_rad', 0.0)
                # Destination-spot box (visual only; never spawned in CARLA). Drawn
                # first so the trajectory lines render on top of it.
                if getattr(car, 'destination_box_world', None) is not None:
                    pts = [project_world_to_topdown(x, y, actor_loc, camera_yaw_rad,
                            width=image.width, height=image.height)
                           for x, y in car.destination_box_world]
                    pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(data, [pts_np], isClosed=True, color=(0, 165, 255), thickness=2)
                if hasattr(car, 'latest_trajectory') and car.latest_trajectory is not None:
                    pts = [project_world_to_topdown(x, y, actor_loc, camera_yaw_rad,
                            width=image.width, height=image.height)
                           for x, y in car.latest_trajectory]
                    pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(data, [pts_np], isClosed=False, color=(0, 255, 0), thickness=2)
                if hasattr(car, 'latest_speed_wps') and car.latest_speed_wps is not None:
                    spd = car.latest_speed_wps
                    spd_pts = [project_world_to_topdown(x, y, actor_loc, camera_yaw_rad,
                                width=image.width, height=image.height)
                               for x, y in spd]
                    # Color each dot by inter-waypoint distance: magenta=slow → red=fast.
                    # Warm tones keep speed wps visually separate from the green route
                    # and cyan A* overlays.
                    dists = np.linalg.norm(np.diff(spd, axis=0), axis=1)
                    max_d = dists.max() if dists.max() > 1e-3 else 1.0
                    for i, (px, py) in enumerate(spd_pts):
                        t = float(dists[i] / max_d) if i < len(dists) else 0.0
                        # BGR: (255,0,255) magenta → (0,0,255) red
                        dot_color = (int(255 * (1 - t)), 0, 255)
                        cv2.circle(data, (px, py), 3, dot_color, -1)
                if hasattr(car, 'latest_astar_trajectory') and car.latest_astar_trajectory is not None:
                    pts = [project_world_to_topdown(x, y, actor_loc, camera_yaw_rad,
                            width=image.width, height=image.height)
                           for x, y in car.latest_astar_trajectory]
                    pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(data, [pts_np], isClosed=False, color=(0, 255, 255), thickness=2)
            
            writer.write(data)
            if car.car.mode == Mode.PARKED:
                car.has_recorded_segment = True
                for _ in range(15):
                    writer.write(data)

def finalize_recording(car):
    for writer_attr, path_attr in [
        ('recording_writer',         'recording_path'),
        ('recording_writer_topdown', 'recording_path_topdown'),
    ]:
        writer = getattr(car, writer_attr, None)
        if not writer:
            continue
        writer.release()
        setattr(car, writer_attr, None)

        path = getattr(car, path_attr, None)
        if path and os.path.exists(path):
            tmp = path + '.tmp.mp4'
            os.rename(path, tmp)
            os.system(f'ffmpeg -y -i "{tmp}" -vcodec libx264 -crf 28 -acodec aac "{path}" -loglevel quiet')
            os.remove(tmp)

def prompt_delete(paths):
    existing = [p for p in paths if p and os.path.exists(p)]
    if not existing:
        return
    print("\n--- Recordings complete ---")
    print("The following recordings were saved:")
    for p in existing:
        print(f"  {p}")
    answer = input("Delete ALL recordings? [y/N]: ").strip().lower()
    if answer == 'y':
        for p in existing:
            os.remove(p)
            print(f"Deleted {p}")
    else:
        # Ask individually
        for p in existing:
            answer = input(f"Delete '{p}'? [y/N]: ").strip().lower()
            if answer == 'y':
                os.remove(p)
                print(f"Deleted {p}")
            else:
                print(f"Kept {p}")

def make_cleanup_handler(car_getter, paths):
    def _handler(signum, frame):
        print("\nInterrupt received, finalizing recordings...")
        car = car_getter()
        if car is not None:
            finalize_recording(car)
        prompt_delete(paths)
        sys.exit(0)

    return _handler


# def cleanup(signum, frame):
#         print("\nInterrupt received, finalizing...")
#         prompt_delete(all_recording_paths)
#         sys.exit(0)


def open_recordings(paths):
    existing = [p for p in paths if p and os.path.exists(p)]
    for path in existing:
        subprocess.Popen(['code', path])


def save_trajectory_plot(trajectory_log, out_path):
    """Save a matplotlib plot of accumulated model-predicted trajectories.

    Args:
        trajectory_log: list of (N, 2) arrays in world coords, one per tick.
        out_path: file path for the saved image (e.g. .png).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    n = len(trajectory_log)
    for i, traj in enumerate(trajectory_log):
        alpha = 0.15 + 0.85 * (i / max(n - 1, 1))
        ax.plot(traj[:, 0], traj[:, 1], color='blue', alpha=alpha, linewidth=0.8)
    ax.set_aspect('equal')
    ax.set_title('Model predicted trajectories')
    ax.set_xlabel('x (world)')
    ax.set_ylabel('y (world)')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved trajectory plot to {out_path}')


def ego_to_world(pred_route, ego_x, ego_y, yaw):
    """Convert ego-frame waypoints (forward, left) to world coords."""
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    world = np.empty_like(pred_route)
    world[:, 0] = ego_x + pred_route[:, 0] * cos_y - pred_route[:, 1] * sin_y
    world[:, 1] = ego_y + pred_route[:, 0] * sin_y + pred_route[:, 1] * cos_y
    return world


def project_world_to_topdown(world_x, world_y, actor_loc, camera_yaw_rad=0.0, width=480, height=320, fov=90, cam_z=30):
    """Project world coordinates to top-down camera pixel coordinates.

    camera_yaw_rad: the camera's world yaw (radians). Must match the yaw used
    when spawning/updating the camera so overlays align with image content.
    Set to initial_car_yaw to make the car's forward direction appear up.
    """
    visible_range = 2 * cam_z  # horizontal FOV=90 → 60m at cam_z=30
    ppm = width / visible_range  # pixels per metre, same for both axes

    dx = world_x - actor_loc.x
    dy = world_y - actor_loc.y

    # For pitch=-90 yaw=θ: image-right = (-sinθ, cosθ), image-up = (cosθ, sinθ)
    cos_c, sin_c = np.cos(camera_yaw_rad), np.sin(camera_yaw_rad)
    right_comp = -sin_c * dx + cos_c * dy
    up_comp    =  cos_c * dx + sin_c * dy

    px = int(width/2  + right_comp * ppm)
    py = int(height/2 - up_comp * ppm)
    return px, py