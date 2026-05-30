#!/usr/bin/env python3
"""
Create a train/val directory split based on destination parking spot IDs.

Episodes going to val destinations are symlinked into routes_validation/,
episodes going to train destinations into routes_training/.  The simlingo
dataloader then uses these directories when use_town13=False.

Usage:
    python prepare_destination_split.py \
        --src  finetune/run_001_backup/data/simlingo/parking_ft/routes_training/RouteScenario_parking \
        --out  finetune/run_001_dest_split \
        [--val-frac 0.20] [--seed 42] [--dry-run]

The output directory will be structured as:
    <out>/data/simlingo/parking_ft/
        routes_training/RouteScenario_parking/<episode_dirs>   (train destinations)
        routes_validation/RouteScenario_parking/<episode_dirs> (val destinations)

Add to train.sh:
    data_module.base_dataset.data_path=../finetune/run_001_dest_split
    data_module.base_dataset.use_town13=false
"""

import argparse
import glob
import json
import os
import random
import sys
from collections import defaultdict


def load_meta(episode_dir):
    path = os.path.join(episode_dir, "episode_meta.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Build destination-based train/val split.")
    parser.add_argument(
        "--src",
        default="finetune/run_001_backup/data/simlingo/parking_ft/routes_training/RouteScenario_parking",
        help="Source RouteScenario_parking directory containing Town04_XXXX episode dirs",
    )
    parser.add_argument(
        "--out",
        default="finetune/run_001_dest_split",
        help="Output base directory (relative to repo root, same level as finetune/)",
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.20,
        help="Fraction of unique destinations to hold out for validation (default: 0.20)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without creating any files")
    args = parser.parse_args()

    src = os.path.abspath(args.src)
    if not os.path.isdir(src):
        sys.exit(f"ERROR: source directory not found:\n  {src}")

    # ── discover episodes ──────────────────────────────────────────────────────
    episode_dirs = sorted(
        d for d in glob.glob(os.path.join(src, "Town*"))
        if os.path.isdir(d) and os.path.isdir(os.path.join(d, "rgb"))
    )
    if not episode_dirs:
        sys.exit(f"No episode directories found under:\n  {src}")

    # ── group by destination ───────────────────────────────────────────────────
    by_dest = defaultdict(list)
    missing_meta = 0
    for ep in episode_dirs:
        meta = load_meta(ep)
        dest = meta.get("destination")
        if dest is None:
            missing_meta += 1
            dest = "unknown"
        by_dest[dest].append(ep)

    all_dests = sorted(d for d in by_dest if d != "unknown")
    n_val = max(1, round(len(all_dests) * args.val_frac))

    random.seed(args.seed)
    val_dests = set(random.sample(all_dests, n_val))
    train_dests = set(all_dests) - val_dests

    print(f"Episodes found:        {len(episode_dirs)}")
    print(f"Unique destinations:   {len(all_dests)}  {sorted(all_dests)}")
    if missing_meta:
        print(f"  (missing episode_meta.json: {missing_meta} episodes → bucketed as 'unknown')")
    print(f"Val destinations ({n_val}): {sorted(val_dests)}")
    print(f"Train destinations ({len(train_dests)}): {sorted(train_dests)}")

    train_eps = [ep for d in train_dests for ep in by_dest[d]]
    val_eps   = [ep for d in val_dests   for ep in by_dest[d]]
    unknown_eps = by_dest.get("unknown", [])

    print(f"\nEpisodes → train: {len(train_eps)}, val: {len(val_eps)}, "
          f"no-meta (goes to train): {len(unknown_eps)}")
    train_eps += unknown_eps

    # ── build output directory structure ──────────────────────────────────────
    # Mirrors: <out>/data/simlingo/parking_ft/routes_{train,val}idation/RouteScenario_parking/
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_base  = os.path.join(repo_root, args.out)

    scenario_name = os.path.basename(src)           # RouteScenario_parking
    route_name    = os.path.basename(os.path.dirname(src))  # parking_ft (from routes_training/..)
    # Infer the intermediate path: data/simlingo/<route_name>/
    inner = os.path.join("data", "simlingo", route_name)

    train_dir = os.path.join(out_base, inner, "routes_training",   scenario_name)
    val_dir   = os.path.join(out_base, inner, "routes_validation",  scenario_name)

    print(f"\nOutput root: {out_base}")
    print(f"  routes_training/  → {len(train_eps)} episodes")
    print(f"  routes_validation/ → {len(val_eps)} episodes")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    if os.path.exists(out_base):
        ans = input(f"\nOutput dir already exists: {out_base}\nOverwrite? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return
        import shutil
        shutil.rmtree(out_base)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir,   exist_ok=True)

    def make_symlinks(episodes, target_dir):
        for ep in episodes:
            name = os.path.basename(ep)
            link = os.path.join(target_dir, name)
            if not os.path.exists(link):
                os.symlink(ep, link)

    make_symlinks(train_eps, train_dir)
    make_symlinks(val_eps,   val_dir)

    print("\nDone. Add these overrides to train.sh / eval_destinations.sh:")
    rel_out = os.path.relpath(out_base,
                              os.path.join(repo_root, "simlingo"))
    print(f"  data_module.base_dataset.data_path={rel_out}")
    print(f"  data_module.base_dataset.use_town13=false")


if __name__ == "__main__":
    main()
