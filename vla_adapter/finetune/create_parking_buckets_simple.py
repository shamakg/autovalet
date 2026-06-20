#!/usr/bin/env python3
"""create_parking_buckets_simple.py

Single-pass replacement for the two-stage create_parking_buckets.py (v1) +
create_parking_buckets_v3.py (split) pipeline.  It opens each measurement
exactly once and, in that pass, emits:

  * the phase buckets            (all, approach, swing_out, final_turn,
                                  into_spot, turn_execution, recovery_park,
                                  correction_maneuver, pre_turn)
  * each phase split by scenario (freeroll / ped / door)
  * the background 'all' split   (all_freeroll / all_ped / all_door)
  * turn_execution_ped_moving    (ped turn_execution frames with speed > 0.5)

The output is byte-identical to running the old v1 builder followed by the old
v3 splitter: same keys in the same order, same path list (in scan order) per
bucket.  The phase thresholds and the freeroll/ped/door grouping are copied
verbatim from the originals, so the buckets the BEST model trained on are
reproduced exactly — this only removes the v1->v3 indirection and the redundant
second pass that re-opened episode_meta.json / re-gunzipped measurements.

Usage:
    python create_parking_buckets_simple.py [--data-dir DIR] [--out-dir DIR]
"""

import argparse
import collections
import gzip
import json
import os
import pathlib
import pickle

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
    "/finetune/parking_buckets_v3"
)

# ---- phase classification (verbatim from create_parking_buckets.py) ----------
RECOVERY_TYPES = {"recovery", "pedestrian_recovery"}
NEARTERM_N = 5
LAT_STRAIGHT_THRESH = 0.5
LAT_SWING_THRESH    = 0.5
TX_FAR_THRESH       = 8.0
TX_FINAL_THRESH     = 3.0
TX_INTO_THRESH      = 6.0
TX_TURN_EXEC_THRESH = 8.0
LAT_MID_TURN_THRESH = 1.0
MIDTERM_START = 5
MIDTERM_END   = 15


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
        buckets.append("into_spot")
    if 0 < target_x < TX_TURN_EXEC_THRESH:
        buckets.append("turn_execution")

    lat_mid = float(np.max(np.abs(route[MIDTERM_START:MIDTERM_END, 1])))
    if lat_near < LAT_STRAIGHT_THRESH and lat_mid > LAT_MID_TURN_THRESH and 3 < target_x < 10:
        buckets.append("pre_turn")
    if ep_type in RECOVERY_TYPES:
        buckets.append("recovery_park")
    if abs(target_x) < 8.0 and route[1, 0] < 0:
        buckets.append("correction_maneuver")

    return buckets


# ---- scenario grouping (verbatim from create_parking_buckets_v3.py) ----------
CONE_TYPES  = {"normal", "normal_close", "recovery"}
EMPTY_TYPES = {"normal_empty", "normal_close_empty"}
PED_TYPES   = {"pedestrian_normal", "pedestrian_recovery"}
DOOR_TYPES  = {"door_normal"}
FREEROLL    = CONE_TYPES | EMPTY_TYPES

SPLIT_BUCKETS = (
    "approach",
    "swing_out",
    "final_turn",
    "into_spot",
    "pre_turn",
    "turn_execution",
    "recovery_park",
    "correction_maneuver",
)

# v1 phase buckets, in their original dict-literal insertion order — keeping
# this order makes the resulting pkl byte-identical to the v1+v3 output.
PHASE_KEYS = (
    "all", "approach", "swing_out", "final_turn", "into_spot",
    "turn_execution", "recovery_park", "correction_maneuver", "pre_turn",
)

SPEED_THRESH = 0.5  # m/s — turn_execution_ped_moving cutoff


def group(ep_type: str) -> str:
    if ep_type in FREEROLL:   return "freeroll"
    if ep_type in PED_TYPES:  return "ped"
    if ep_type in DOOR_TYPES: return "door"
    return "unk"


def rel_path(abs_path: pathlib.Path) -> str:
    return pathlib.Path(os.path.relpath(abs_path, REPO_ROOT)).as_posix()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir",  type=pathlib.Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise SystemExit(f"Data dir not found: {args.data_dir}")

    # Pre-insert every key in the same order the v1+v3 pipeline produced, so
    # the pkl is byte-identical (dicts pickle in insertion order).
    buckets: dict[str, list[str]] = {k: [] for k in PHASE_KEYS}
    for base in SPLIT_BUCKETS:
        for suffix in ("freeroll", "ped", "door"):
            buckets[f"{base}_{suffix}"] = []
    for suffix in ("freeroll", "ped", "door"):
        buckets[f"all_{suffix}"] = []
    buckets["turn_execution_ped_moving"] = []

    split_set = set(SPLIT_BUCKETS)
    episodes = sorted(args.data_dir.glob("Town04_*"))
    print(f"Scanning {len(episodes)} episodes in {args.data_dir}")

    skipped = 0
    for ep_dir in episodes:
        meta_path = ep_dir / "episode_meta.json"
        meas_dir = ep_dir / "measurements"
        if not meta_path.exists() or not meas_dir.exists():
            skipped += 1
            continue

        ep_type = json.load(open(meta_path)).get("episode_type", "unknown")
        g = group(ep_type)                       # per-episode: all frames share it
        is_split_group = g in ("freeroll", "ped", "door")

        for meas_path in sorted(meas_dir.glob("*.json.gz")):
            with gzip.open(meas_path, "rt") as fh:
                d = json.load(fh)
            route    = np.array(d["route"])      # (20, 2)
            target_x = float(d["target_point"][0])
            speed    = float(d.get("speed", 0.0))

            rel = rel_path(meas_path)
            phases = classify(route, target_x, ep_type)
            for b in phases:
                buckets[b].append(rel)
                if b in split_set and is_split_group:
                    buckets[f"{b}_{g}"].append(rel)
            if is_split_group:                    # 'all' split (every frame)
                buckets[f"all_{g}"].append(rel)
            if "turn_execution" in phases and g == "ped" and speed > SPEED_THRESH:
                buckets["turn_execution_ped_moving"].append(rel)

    total = len(buckets["all"])
    print(f"\nScanned {total} frames ({skipped} episodes skipped — no meta/measurements)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_pkl = args.out_dir / "buckets_paths.pkl"
    with open(out_pkl, "wb") as fh:
        pickle.dump(buckets, fh)
    print(f"Saved bucket pkl → {out_pkl}")

    # Stats in the v3 format (informational; the datamodule only reads the pkl).
    grp_of: dict[str, str] = {}
    for ep_dir in episodes:
        mp = ep_dir / "episode_meta.json"
        if mp.exists():
            grp_of[ep_dir.name] = group(json.load(open(mp)).get("episode_type", "unk"))

    def gof(rel: str) -> str:
        # rel = ../finetune/.../RouteScenario_parking/<EP>/measurements/<f>.json.gz
        return grp_of.get(pathlib.PurePosixPath(rel).parents[1].name, "unk")

    stats = {}
    for name in sorted(buckets):
        gc = collections.Counter(gof(r) for r in buckets[name])
        stats[name] = {"total": len(buckets[name]), "freeroll": gc["freeroll"],
                       "ped": gc["ped"], "door": gc["door"], "unk": gc["unk"]}
    json.dump(stats, open(args.out_dir / "buckets_stats.json", "w"), indent=2)
    print(f"Saved stats → {args.out_dir / 'buckets_stats.json'}")


if __name__ == "__main__":
    main()
