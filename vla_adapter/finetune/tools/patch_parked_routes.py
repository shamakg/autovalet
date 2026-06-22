#!/usr/bin/env python3
"""patch_parked_routes.py

Zeros the 'route' field in measurement files at the end of each episode
where the car is in the parking spot (target_x < TX_THRESH), then adds
those frames to parking_buckets_v2 as a 'parked' bucket.

Why in-place edits rather than synthetic episodes:
  The dataset loader requires at least 23 frames per episode (skip_first=10
  + pred_len=11 + hist_len=1 + 1).  Editing existing multi-frame episodes
  avoids that windowing constraint entirely.

What is zeroed:
  route only — [[0,0] x 20].
  route_original is left intact so we can always recover the original labels.
  The speed_wps (waypoints) label is derived from future ego_matrix positions,
  which are already near-zero because the car is stationary at episode end.

Usage:
    # Dry run — print what would be patched, touch nothing
    python patch_parked_routes.py --dry-run

    # Apply patches + update bucket pkl
    python patch_parked_routes.py
"""

import argparse, gzip, json, pathlib, pickle, collections
import numpy as np

REPO_ROOT = pathlib.Path(
    "/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo"
).resolve()

DEFAULT_DATA = pathlib.Path(
    "/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter"
    "/finetune/run_001/data/simlingo/parking_ft/routes_training/RouteScenario_parking"
)
DEFAULT_V2 = pathlib.Path(
    "/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter"
    "/finetune/parking_buckets_v2"
)

TX_THRESH  = 0.5    # m — only patch frames this close to destination
ZERO_ROUTE = [[0.0, 0.0]] * 20


def rel_path(abs_path: pathlib.Path) -> str:
    return pathlib.Path(
        __import__('os').path.relpath(abs_path, REPO_ROOT)
    ).as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA)
    parser.add_argument("--v2-dir",   type=pathlib.Path, default=DEFAULT_V2)
    parser.add_argument("--thresh",   type=float, default=TX_THRESH,
                        help="target_x threshold below which frames are patched")
    parser.add_argument("--dry-run",  action="store_true",
                        help="print what would be patched without writing anything")
    args = parser.parse_args()

    episodes = sorted(args.data_dir.glob("Town04_*"))
    print(f"Scanning {len(episodes)} episodes  (tx_thresh={args.thresh}m, dry_run={args.dry_run})")

    parked_paths = []
    stats = collections.Counter()
    ep_type_counts = collections.Counter()
    dest_counts = collections.Counter()

    for ep_dir in episodes:
        meas_dir = ep_dir / "measurements"
        meta_path = ep_dir / "episode_meta.json"
        if not meas_dir.exists() or not meta_path.exists():
            stats["skipped"] += 1
            continue

        meta = json.load(open(meta_path))
        ep_type = meta.get("episode_type", "unk")
        dest    = meta.get("destination", -1)

        meas_files = sorted(meas_dir.glob("*.json.gz"))
        if not meas_files:
            stats["skipped"] += 1
            continue

        # Walk backwards from end, patch while close to destination
        patched_this_ep = 0
        for meas_path in reversed(meas_files):
            d = json.load(gzip.open(meas_path, "rt"))
            tx = float(d["target_point"][0])

            if tx > args.thresh:
                break   # further from destination — stop walking back

            already_zero = (np.max(np.abs(d["route"])) < 1e-6)
            if already_zero:
                # Already patched (e.g. from a previous run)
                parked_paths.append(rel_path(meas_path))
                patched_this_ep += 1
                continue

            if not args.dry_run:
                d["route"] = ZERO_ROUTE
                with gzip.open(meas_path, "wt") as fh:
                    json.dump(d, fh)

            parked_paths.append(rel_path(meas_path))
            patched_this_ep += 1
            stats["patched"] += 1

        if patched_this_ep:
            stats["episodes_with_patches"] += 1
            ep_type_counts[ep_type] += patched_this_ep
            dest_counts[dest] += patched_this_ep

    # ------------------------------------------------------------------
    print(f"\nPatched {stats['patched']} frames across "
          f"{stats['episodes_with_patches']} episodes "
          f"({'DRY RUN — no files written' if args.dry_run else 'files written'})")
    print(f"Total parked bucket size: {len(parked_paths)} frames")

    print(f"\nBy episode type:")
    for et, cnt in sorted(ep_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {et:<25}: {cnt:4d} frames")

    print(f"\nBy destination:")
    for dest in sorted(dest_counts):
        print(f"  dest={dest:>3}: {dest_counts[dest]:3d} frames")

    if args.dry_run:
        print("\n(dry run — bucket pkl not updated)")
        return

    # ------------------------------------------------------------------
    # Update v2 pkl
    v2_pkl = args.v2_dir / "buckets_paths.pkl"
    print(f"\nLoading v2 pkl from {v2_pkl}")
    v2 = pickle.load(open(v2_pkl, "rb"))
    v2["parked"] = parked_paths
    pickle.dump(v2, open(v2_pkl, "wb"))
    print(f"Updated 'parked' bucket ({len(parked_paths)} frames) → saved v2 pkl")

    print(f"\nSuggested yaml entry:")
    print(f"    parked: 0.40   # {len(parked_paths)} frames — "
          f"end-of-episode frames with route zeroed (tx < {args.thresh}m)")


if __name__ == "__main__":
    main()
