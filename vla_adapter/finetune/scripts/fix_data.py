"""Post-process run_001 training data.

Fixes:
  1. target_point: changed from 1.6m-offset A* destination to geometric spot center.
  2. lmdrive_command: distances recomputed from the corrected target_point.
  3. Removes entire episodes with IoU <= threshold (default 0.7).

Usage:
    python scripts/fix_data.py [--dry-run] [--min-iou 0.7]
"""

import argparse
import gzip
import json
import pathlib
import random
import shutil
import sys

import numpy as np
import ujson

sys.path.insert(0, '/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet')
from parking_position import parking_vehicle_locations_Town04 as SPOTS

sys.path.insert(0, '/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo')
from team_code.transfuser_utils import inverse_conversion_2d

_TEMPLATE_PATH = '/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo/data/augmented_templates/lmdrive.json'
with open(_TEMPLATE_PATH) as _f:
    _CMD_TEMPLATES = json.load(_f)

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

RUN_DIR = pathlib.Path("/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/finetune/run_001")
EPISODE_BASE = RUN_DIR / "data/simlingo/parking_ft/routes_training/RouteScenario_parking"


def make_command(target_pt):
    target_x = round(abs(target_pt[0]), 1)
    target_y = round(abs(target_pt[1]), 1)
    is_left = target_pt[1] > 0
    is_behind = target_pt[0] < 0
    if is_behind:
        candidates = _REVERSE_TEMPLATES['left' if is_left else 'right']
    else:
        candidates = [t for t in _CMD_TEMPLATES["65"]
                      if not ("left" in t and not is_left)
                      and not ("right" in t and is_left)]
    return random.choice(candidates).replace("[x]", str(target_x)).replace("[y]", str(target_y))


def fix_episode(ep_dir, dry_run=False):
    meta_path = ep_dir / "episode_meta.json"
    if not meta_path.exists():
        return 0, 0

    with open(meta_path) as f:
        meta = json.load(f)
    spot_idx = meta.get("destination", -1)
    if spot_idx < 0 or spot_idx >= len(SPOTS):
        print(f"  WARNING: bad spot_idx {spot_idx} in {ep_dir.name}")
        return 0, 0

    spot = SPOTS[spot_idx]
    spot_xy = np.array([spot.x, spot.y])

    frames = sorted((ep_dir / "measurements").glob("*.json.gz"))
    fixed = 0

    for f in frames:
        with gzip.open(f, "rt") as fp:
            m = ujson.load(fp)

        ego_M = np.array(m["ego_matrix"])
        ego_xy = ego_M[:2, 3]
        yaw = np.arctan2(ego_M[1, 0], ego_M[0, 0])

        new_tp = inverse_conversion_2d(spot_xy, ego_xy, yaw).tolist()

        m["target_point"] = new_tp
        m["target_point_next"] = new_tp
        m["lmdrive_command"] = make_command(new_tp)

        if not dry_run:
            with gzip.open(f, "wt", encoding="utf-8") as fp:
                ujson.dump(m, fp, indent=4)
        fixed += 1

    return len(frames), fixed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-iou", type=float, default=0.7)
    args = parser.parse_args()

    episodes = sorted([p.parent for p in EPISODE_BASE.glob("**/results.json.gz")])
    print(f"Episodes: {len(episodes)}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"Min IoU: {args.min_iou}\n")

    total_frames = 0
    total_fixed = 0
    episodes_removed = 0
    frames_removed = 0

    for i, ep in enumerate(episodes):
        with open(ep / "episode_meta.json") as f:
            meta = json.load(f)
        iou = meta.get("iou", 0)

        n_frames = len(list((ep / "measurements").glob("*.json.gz")))

        if iou <= args.min_iou:
            if not args.dry_run:
                shutil.rmtree(ep)
            episodes_removed += 1
            frames_removed += n_frames
            continue

        n, fixed = fix_episode(ep, dry_run=args.dry_run)
        total_frames += n
        total_fixed += fixed
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(episodes)}] processed...")

    print(f"\nDone.")
    print(f"  Episodes removed (IoU <= {args.min_iou}): {episodes_removed}/{len(episodes)}")
    print(f"  Frames removed:  {frames_removed}")
    print(f"  Frames fixed:    {total_fixed}")
    print(f"  Frames remaining:{total_fixed}")


if __name__ == "__main__":
    main()
