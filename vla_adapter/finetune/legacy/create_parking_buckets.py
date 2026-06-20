#!/usr/bin/env python3
"""create_parking_buckets.py

Scans parking training data and classifies frames into 5 buckets:

  all         — every frame (background diversity)
  approach    — near-term path is straight AND far from spot
                  max(|route[:5, 1]|) < 0.5 m  AND  target_x > 8.0 m
  swing_out   — near-term path is curving AND not yet at spot
                  max(|route[:5, 1]|) > 0.5 m  AND  target_x > 3.0 m
  final_turn  — close to destination
                  target_x < 3.0 m
  recovery    — episode collected in recovery or pedestrian_recovery mode
                  episode_meta["episode_type"] in {"recovery", "pedestrian_recovery"}

Frames can belong to multiple buckets (e.g. a recovery frame that is also a
final_turn). The datamodule handles this by instantiating separate weighted
datasets for each bucket.

Output
------
  <out_dir>/buckets_paths.pkl   — Dict[str, List[str]] of measurement paths
  <out_dir>/buckets_stats.json  — per-bucket frame counts

The measurement paths are stored relative to the simlingo repo root so that
dataset_base.py's path reconstruction logic produces correct absolute paths:
  "{repo_path}/{run_id_parent}" == str(measurement_file_path.parent)

Usage
-----
  python create_parking_buckets.py [--data-dir DIR] [--out-dir DIR]
"""

import argparse
import gzip
import json
import pathlib
import pickle

import os

import numpy as np


REPO_ROOT = pathlib.Path(
    "/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo"
).resolve()

DEFAULT_DATA = pathlib.Path(
    "/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter"
    "/finetune/run_001/data/simlingo/parking_ft/routes_training/RouteScenario_parking"
)

DEFAULT_OUT = pathlib.Path(
    "/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter"
    "/finetune/parking_buckets_v1"
)

RECOVERY_TYPES = {"recovery", "pedestrian_recovery"}

# Near-term window: first N waypoints used to judge current path curvature
NEARTERM_N = 5
LAT_STRAIGHT_THRESH = 0.5   # m — below this = straight near-term path
LAT_SWING_THRESH    = 0.5   # m — above this = curving near-term path
TX_FAR_THRESH       = 8.0   # m — above this = far from spot
TX_FINAL_THRESH     = 3.0   # m — below this = near spot
TX_INTO_THRESH      = 6.0   # m — into-spot: turned and driving straight in
TX_TURN_EXEC_THRESH = 8.0   # m — turn_execution: full 0-8m window (mid-turn + pre-turn)
LAT_MID_TURN_THRESH = 1.0   # m — mid-term lateral deviation indicating turn is imminent
MIDTERM_START = 5           # waypoint index where mid-term window begins
MIDTERM_END   = 15          # waypoint index where mid-term window ends


def classify(route: np.ndarray, target_x: float, ep_type: str) -> list[str]:
    buckets = ["all"]
    lat_near = float(np.max(np.abs(route[:NEARTERM_N, 1])))

    if lat_near < LAT_STRAIGHT_THRESH and target_x > TX_FAR_THRESH:
        buckets.append("approach")
    if lat_near > LAT_SWING_THRESH and target_x > TX_FINAL_THRESH:
        buckets.append("swing_out")
    if target_x < TX_FINAL_THRESH:
        buckets.append("final_turn")
    if lat_near < LAT_STRAIGHT_THRESH and 0 < target_x < TX_INTO_THRESH:
        # Car has already turned and is driving straight into the space:
        # near-term route is straight (turn complete) and destination is close ahead.
        buckets.append("into_spot")
    if 0 < target_x < TX_TURN_EXEC_THRESH:
        # Full 0-8 m window regardless of curvature: covers both the pre-turn
        # approach (where the model should be anticipating the turn) and the
        # mid-turn execution (where curvature is high). Oversampling this teaches
        # the model to commit to the turn at the right distance.
        buckets.append("turn_execution")

    lat_mid = float(np.max(np.abs(route[MIDTERM_START:MIDTERM_END, 1])))
    if lat_near < LAT_STRAIGHT_THRESH and lat_mid > LAT_MID_TURN_THRESH and 3 < target_x < 10:
        # Pre-turn anticipation: near-term path is still straight but the mid-term
        # route already shows the turn beginning. These frames teach the model exactly
        # when to initiate the turn — the critical "trigger" signal.
        buckets.append("pre_turn")
    if ep_type in RECOVERY_TYPES:
        # Named 'recovery_park' (not 'recovery') so SimLingo's dataset loader uses
        # its generic bucket lookup instead of the hardcoded recovery_data_small/large case.
        buckets.append("recovery_park")

    if abs(target_x) < 8.0 and route[1, 0] < 0:
        # Reversing while near the spot: covers both overshoot corrections (tx < 0)
        # and near-spot repositioning (0 < tx < 8). Not labeled explicitly in episode
        # meta — identified purely from route direction and proximity.
        buckets.append("correction_maneuver")

    return buckets


def rel_path(abs_path: pathlib.Path) -> str:
    """Return path relative to REPO_ROOT as a POSIX string.

    Uses os.path.relpath (not Path.relative_to) because the measurement files
    live outside REPO_ROOT (they're at ../finetune/...). The resulting
    '../finetune/...' strings match what dataset_base.py builds via
    f"{repo_path}/{run_id_parent}" when glob preserves the '..' in its results.
    """
    return pathlib.Path(os.path.relpath(abs_path, REPO_ROOT)).as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA,
                        help="RouteScenario_parking directory containing Town04_* episodes")
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT,
                        help="Output directory for pkl + stats")
    args = parser.parse_args()

    data_dir: pathlib.Path = args.data_dir
    out_dir:  pathlib.Path = args.out_dir

    if not data_dir.exists():
        raise SystemExit(f"Data dir not found: {data_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, list[str]] = {
        "all": [], "approach": [], "swing_out": [], "final_turn": [],
        "into_spot": [], "turn_execution": [], "recovery_park": [],
        "correction_maneuver": [], "pre_turn": [],
    }

    episodes = sorted(data_dir.glob("Town04_*"))
    print(f"Scanning {len(episodes)} episodes in {data_dir}")

    skipped = 0
    for ep_dir in episodes:
        meta_path = ep_dir / "episode_meta.json"
        if not meta_path.exists():
            skipped += 1
            continue

        with open(meta_path) as fh:
            meta = json.load(fh)
        ep_type = meta.get("episode_type", "unknown")

        meas_dir = ep_dir / "measurements"
        if not meas_dir.exists():
            skipped += 1
            continue

        for meas_path in sorted(meas_dir.glob("*.json.gz")):
            with gzip.open(meas_path, "rt") as fh:
                d = json.load(fh)

            route    = np.array(d["route"])          # (20, 2)
            target_x = float(d["target_point"][0])

            frame_buckets = classify(route, target_x, ep_type)

            # Store path relative to simlingo repo root
            rel = rel_path(meas_path)
            for b in frame_buckets:
                buckets[b].append(rel)

    total = len(buckets["all"])
    print(f"\nScanned {total} frames ({skipped} episodes skipped — no meta/measurements)")
    print("\nBucket distribution:")
    stats = {}
    for name, paths in buckets.items():
        pct = 100 * len(paths) / max(total, 1)
        print(f"  {name:12s}: {len(paths):6d}  ({pct:.1f}%)")
        stats[name] = {"count": len(paths), "pct": round(pct, 1)}

    pkl_path = out_dir / "buckets_paths.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump(buckets, fh)
    print(f"\nSaved bucket pkl → {pkl_path}")

    stats_path = out_dir / "buckets_stats.json"
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"Saved bucket stats → {stats_path}")


if __name__ == "__main__":
    main()
