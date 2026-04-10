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


def init_recording(car, recording_path, width=480, height=320, fps=20, top_down=False):
        world = car.world
        actor = car.actor
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(width))
        cam_bp.set_attribute('image_size_y', str(height))
        cam_bp.set_attribute('fov', str(90))
        if top_down:
            cam_transform = carla.Transform(
                carla.Location(x=0, z=30),
                carla.Rotation(pitch=-90)
            )
            frames_attr = 'frames_topdown'
            writer_attr = 'recording_writer_topdown'
            path_attr   = 'recording_path_topdown'
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
            if frames_attr == 'frames_topdown' and hasattr(car, 'latest_trajectory') and car.latest_trajectory is not None:
                actor_loc = car.actor.get_location()
                pts = [project_world_to_topdown(x, y, actor_loc,
                        width=image.width, height=image.height)
                       for x, y in car.latest_trajectory]
                pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(data, [pts_np], isClosed=False, color=(0, 255, 0), thickness=2)
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
            os.system(f'ffmpeg -y -i "{tmp}" -vcodec libx264 -crf 18 -acodec aac "{path}" -loglevel quiet')
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


def project_world_to_topdown(world_x, world_y, actor_loc, width=480, height=320, fov=90, cam_z=30):
    """Project world coordinates to top-down camera pixel coordinates."""
    # Camera is at actor location + (0, 0, 30), looking straight down
    cam_x = actor_loc.x
    cam_y = actor_loc.y
    
    # With fov=90 and pitch=-90, the visible range at ground level is:
    # visible_range = 2 * cam_z * tan(fov/2) = 2 * 30 * tan(45°) = 60m
    visible_range = 2 * cam_z  # = 60m
    
    # pixels per meter
    ppm_x = width / visible_range
    ppm_y = height / visible_range
    
    # offset from camera center in world coords
    dx = world_x - cam_x
    dy = world_y - cam_y
    
    # convert to pixel (center of image = actor location)
    px = int(width/2 - dx * ppm_x)
    py = int(height/2 - dy * ppm_y)
    
    return px, py