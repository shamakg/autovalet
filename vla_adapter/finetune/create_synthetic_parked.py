#!/usr/bin/env python3
"""create_synthetic_parked.py

Creates a 'parked' training bucket by taking the last frame of every
successful episode and zeroing its route labels.

Why:  The controller predicts ~1.2 m of forward motion even at the final
      frame (target_x ≈ 0.16 m average).  Without correction the model
      learns "go 1.2 m more" when visually in the spot — exactly the
      "doesn't know when parked" failure mode.

What it does:
  1. Reads every Town04_*/measurements/<last>.json.gz
  2. Writes a patched copy to synthetic_parked/<ep>/measurements/<frame>.json.gz
     with route zeroed to [[0,0]×20] and route_original preserved unchanged
  3. Adds these paths to parking_buckets_v2 as a new 'parked' bucket
  4. Prints a full composition report

The rgb/depth images live alongside the measurements so the dataset loader
finds them automatically (same relative layout as real episodes).
"""

import argparse, collections, gzip, json, pathlib, pickle, shutil
import numpy as np

REPO_ROOT = pathlib.Path(
    "/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo"
).resolve()

DEFAULT_DATA = pathlib.Path(
    "/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter"
    "/finetune/run_001/data/simlingo/parking_ft/routes_training/RouteScenario_parking"
)
DEFAULT_SYNTH = pathlib.Path(
    "/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter"
    "/finetune/synthetic_parked/RouteScenario_parking"
)
DEFAULT_V2 = pathlib.Path(
    "/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter"
    "/finetune/parking_buckets_v2"
)

ZERO_ROUTE = [[0.0, 0.0]] * 20


def rel_path(abs_path: pathlib.Path) -> str:
    return pathlib.Path(
        __import__('os').path.relpath(abs_path, REPO_ROOT)
    ).as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir",  type=pathlib.Path, default=DEFAULT_DATA)
    parser.add_argument("--synth-dir", type=pathlib.Path, default=DEFAULT_SYNTH)
    parser.add_argument("--v2-dir",    type=pathlib.Path, default=DEFAULT_V2)
    args = parser.parse_args()

    args.synth_dir.mkdir(parents=True, exist_ok=True)

    episodes = sorted(args.data_dir.glob("Town04_*"))
    print(f"Scanning {len(episodes)} episodes …")

    parked_paths = []
    stats = collections.Counter()

    for ep_dir in episodes:
        meas_dir = ep_dir / "measurements"
        meta_path = ep_dir / "episode_meta.json"
        if not meas_dir.exists() or not meta_path.exists():
            stats["skipped_no_data"] += 1
            continue

        meas_files = sorted(meas_dir.glob("*.json.gz"))
        if not meas_files:
            stats["skipped_empty"] += 1
            continue

        last_meas = meas_files[-1]
        d = json.load(gzip.open(last_meas, "rt"))

        # Sanity check: must be close to destination
        tx = float(d["target_point"][0])
        if tx > 1.0:
            # Episode may not have succeeded — skip
            stats["skipped_not_parked"] += 1
            continue

        # Build synthetic episode directory (mirror layout so image symlinks work)
        synth_ep = args.synth_dir / ep_dir.name
        synth_meas = synth_ep / "measurements"
        synth_meas.mkdir(parents=True, exist_ok=True)

        # Symlink rgb/depth image dirs rather than copying (they can be large)
        for subdir in ("rgb", "rgb_augmented", "depth", "semantics", "lidar",
                       "bev_semantics", "bev_dynamic", "bev_label"):
            src = ep_dir / subdir
            dst = synth_ep / subdir
            if src.exists() and not dst.exists():
                dst.symlink_to(src.resolve())

        # Copy episode_meta.json unchanged (dataset loader needs it)
        shutil.copy2(meta_path, synth_ep / "episode_meta.json")

        # Write patched measurement: zero the route
        d_patched = dict(d)
        d_patched["route"] = ZERO_ROUTE
        # keep route_original so the loader can still access the real route if needed

        out_meas = synth_meas / last_meas.name
        with gzip.open(out_meas, "wt") as fh:
            json.dump(d_patched, fh)

        parked_paths.append(rel_path(out_meas))
        stats["created"] += 1

    print(f"\nCreated {stats['created']} synthetic parked frames")
    print(f"Skipped: {stats['skipped_no_data']} no-data, "
          f"{stats['skipped_empty']} empty, "
          f"{stats['skipped_not_parked']} not-parked (tx>1m)")

    # ------------------------------------------------------------------
    # Load v2 pkl and add 'parked' bucket
    v2_pkl = args.v2_dir / "buckets_paths.pkl"
    print(f"\nLoading v2 pkl from {v2_pkl}")
    v2 = pickle.load(open(v2_pkl, "rb"))
    v2["parked"] = parked_paths
    pickle.dump(v2, open(v2_pkl, "wb"))
    print(f"Added 'parked' bucket ({len(parked_paths)} frames) → saved v2 pkl")

    # ------------------------------------------------------------------
    # Per-episode-type breakdown of the parked bucket
    ep_type_counts = collections.Counter()
    for rel in parked_paths:
        abs_p = REPO_ROOT / rel
        # synthetic path: .../synthetic_parked/RouteScenario_parking/Town04_XXXX/...
        # real meta: .../run_001/.../Town04_XXXX/episode_meta.json
        ep_name = abs_p.parent.parent.name   # Town04_XXXX
        real_meta = args.data_dir / ep_name / "episode_meta.json"
        if real_meta.exists():
            ep_type = json.load(open(real_meta)).get("episode_type", "unk")
        else:
            ep_type = "unk"
        ep_type_counts[ep_type] += 1

    print(f"\nParked bucket breakdown by episode type:")
    for ep_type, count in sorted(ep_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {ep_type:<25}: {count:4d} frames")

    # Per-destination
    dest_counts = collections.Counter()
    for rel in parked_paths:
        abs_p = REPO_ROOT / rel
        ep_name = abs_p.parent.parent.name
        real_meta = args.data_dir / ep_name / "episode_meta.json"
        if real_meta.exists():
            dest = json.load(open(real_meta)).get("destination", -1)
            dest_counts[dest] += 1

    print(f"\nParked bucket breakdown by destination:")
    for dest in sorted(dest_counts):
        print(f"  dest={dest:>3}: {dest_counts[dest]:3d} frames")

    print(f"\nSuggested yaml addition:")
    print(f"    parked: 0.40   # {len(parked_paths)} synthetic frames "
          f"(last frame of each ep, route zeroed)")


if __name__ == "__main__":
    main()
