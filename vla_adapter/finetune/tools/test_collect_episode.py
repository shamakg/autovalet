"""
Quick smoke test for collect_episode.
Run from autovalet/:
    source ~/envs/simlingo/bin/activate
    python vla_adapter/finetune/test_collect_episode.py
"""
import sys, os
for p in [
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    '/home/sumesh/opt/carla/PythonAPI/carla',
    '/home/sumesh/carla_garage/scenario_runner',
    '/home/sumesh/carla_garage/leaderboard',
    '/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo',
]:
    if p not in sys.path:
        sys.path.insert(0, p)

import gzip, ujson, pathlib, shutil
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from testbed.v2_experiment_utils import load_client, town04_load
from vla_adapter.finetune.collect_data import collect_episode

SAVE_PATH = pathlib.Path('/tmp/test_collect_episode')
if SAVE_PATH.exists():
    shutil.rmtree(SAVE_PATH)

client = load_client()
world  = town04_load(client)
CarlaDataProvider.set_client(client)
CarlaDataProvider.set_world(world)

print("Running collect_episode (normal, dest=22)...")
success = collect_episode(world, SAVE_PATH, 'normal', destination=22, parked_spots=[21, 23])

print(f"\n--- Results ---")
print(f"Success: {success}")

rgb_files  = sorted(SAVE_PATH.glob('rgb/*.jpg'))         if SAVE_PATH.exists() else []
meas_files = sorted(SAVE_PATH.glob('measurements/*.json.gz')) if SAVE_PATH.exists() else []
box_files  = sorted(SAVE_PATH.glob('boxes/*.json.gz'))   if SAVE_PATH.exists() else []

print(f"RGB frames    : {len(rgb_files)}")
print(f"Measurements  : {len(meas_files)}")
print(f"Boxes         : {len(box_files)}")

if success and meas_files:
    with gzip.open(meas_files[0], 'rt') as f:
        m = ujson.load(f)
    print(f"\nSample measurement keys: {list(m.keys())}")
    print(f"  speed          : {m['speed']:.2f}")
    print(f"  target_point   : {m['target_point']}")
    print(f"  command        : {m['command']}")
    print(f"  lmdrive_command: {m['lmdrive_command']}")

if success and box_files:
    with gzip.open(box_files[0], 'rt') as f:
        b = ujson.load(f)
    print(f"\nBoxes in first frame: {len(b)}")
    if b:
        print(f"  First box: {b[0]}")

if success:
    meta_path = SAVE_PATH / 'episode_meta.json'
    if meta_path.exists():
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"\nEpisode meta: {meta}")
