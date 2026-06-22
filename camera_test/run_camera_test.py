"""
Quick test: spawn camera per collect_data.setup_camera, capture one frame,
save to camera_test/test_frame.jpg, and report resolution.

Run from autovalet/:
    source ~/envs/simlingo/bin/activate
    python camera_test/run_camera_test.py
"""

import sys, os
AUTOVALET = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [AUTOVALET,
          '/home/sumesh/opt/carla/PythonAPI/carla',
          '/home/sumesh/carla_garage/scenario_runner',
          '/home/sumesh/carla_garage/leaderboard']:
    if p not in sys.path:
        sys.path.insert(0, p)

import queue
import numpy as np
from PIL import Image
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from testbed.v2_experiment_utils import load_client, town04_load
from parking_scenarios.parking_scenario_easy import ParkingScenarioEasy
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_frame.jpg")


def setup_camera(world, vehicle):
    """Mirror of collect_data.setup_camera."""
    bp = world.get_blueprint_library().find('sensor.camera.rgb')
    bp.set_attribute('image_size_x', '1024')
    bp.set_attribute('image_size_y', '512')
    bp.set_attribute('fov', '110')
    transform = carla.Transform(
        carla.Location(x=-1.5, y=0.0, z=2.0),
        carla.Rotation(roll=0.0, pitch=0.0, yaw=0.0)
    )
    return world.spawn_actor(bp, transform, attach_to=vehicle)


def make_config():
    cfg = ScenarioConfiguration()
    cfg.name  = "Parking"
    cfg.type  = "Parking"
    cfg.town  = "Town04_Opt"
    cfg.other_actors = None
    cfg.route = False
    return cfg


def main():
    client = load_client()
    world  = town04_load(client)

    CarlaDataProvider.set_client(client)
    CarlaDataProvider.set_world(world)

    scenario = None
    camera   = None
    try:
        scenario = ParkingScenarioEasy(
            world=world, config=make_config(),
            destination=22, parked=[21, 23],
            criteria_enable=False,
        )
        world.tick()
        world.tick()

        frame_q = queue.Queue(maxsize=1)
        camera = setup_camera(world, scenario.car.actor)
        camera.listen(lambda img: frame_q.put(img) if frame_q.empty() else None)

        # tick until we get a frame (usually 1-2 ticks)
        for _ in range(10):
            world.tick()
            if not frame_q.empty():
                break

        if frame_q.empty():
            print("ERROR: no frame received after 10 ticks")
            return

        img = frame_q.get()
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
        rgb = arr[:, :, :3][..., ::-1]  # BGRA → RGB

        pil = Image.fromarray(rgb)
        pil.save(OUT_PATH, quality=95)

        print(f"Saved: {OUT_PATH}")
        print(f"Resolution: {img.width}x{img.height}")
        print(f"Expected 900x256: {'YES' if img.width == 900 and img.height == 256 else 'NO — got ' + str(img.width) + 'x' + str(img.height)}")

    finally:
        if camera:
            camera.stop()
            camera.destroy()
        if scenario:
            try:
                scenario.cleanup()
            except Exception as e:
                print(f"cleanup: {e}")
        for _ in range(3):
            world.tick()


if __name__ == "__main__":
    main()
