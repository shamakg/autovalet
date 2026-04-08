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