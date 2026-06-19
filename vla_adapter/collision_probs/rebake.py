#!/usr/bin/env python3
"""rebake.py

Regenerate heatmap/ and topdown_heatmap/ for already-collected episodes,
OFFLINE (no CARLA). rgb/ is left untouched (plain front camera) -- the new
two-view pipeline feeds the model rgb/ + topdown_heatmap/ as separate tiles.

Default mode recomputes risk from measurements (picks up the cv2-resize /
corridor-clip / CORRIDOR_SAFE_FLOOR fixes); the per-frame loop mirrors
collect_data_topdown._compute_heatmaps but stays import-light so it runs without
CARLA. --reuse-heatmaps instead builds topdown_heatmap/ from the existing
heatmap/ PNGs without recomputing risk.

Usage:
    python rebake.py --all
    python rebake.py Town04_0009
    python rebake.py --all --reuse-heatmaps
"""
import argparse
import glob
import gzip
import json
import os
import pathlib
import sys

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from crossing_gmm import render_route_risk_grid, heatmap_on_topdown
from config import (HORIZON_SCALE, ROUTE_CORRIDOR_HALF_WIDTH,
                    TOPDOWN_HEIGHT, TOPDOWN_SIZE, TOPDOWN_FOV)

TOPDOWN_HALF = TOPDOWN_HEIGHT * np.tan(np.deg2rad(TOPDOWN_FOV / 2.0))
TOPDOWN_RES = 1.0   # matches collect_data_topdown

_DEFAULT_ROOT = (
    pathlib.Path(_HERE) / "run_topdown_001" / "data" / "simlingo" / "parking_ft"
    / "routes_training" / "RouteScenario_parking"
)


def rebake_episode(ep, reuse_heatmaps=False):
    ep = pathlib.Path(ep)
    hm_dir = ep / "heatmap"
    tdh_dir = ep / "topdown_heatmap"
    tdh_dir.mkdir(exist_ok=True)
    hm_dir.mkdir(exist_ok=True)

    if reuse_heatmaps:
        n = 0
        for hp in sorted(glob.glob(str(hm_dir / "*.png"))):
            stem = pathlib.Path(hp).stem
            tp = ep / "topdown" / f"{stem}.jpg"
            if not tp.exists():
                continue
            risk = np.array(Image.open(hp).convert("L"))
            td = np.array(Image.open(tp).convert("RGB"))
            Image.fromarray(heatmap_on_topdown(td, risk)).save(tdh_dir / f"{stem}.jpg", quality=90)
            n += 1
        print(f"  {ep.name}: rebuilt {n} topdown_heatmap frames (reuse-heatmaps)")
        return

    # Full regen from measurements (mirrors collect_data_topdown._compute_heatmaps).
    base = []
    gp = ep / "gmm.json"
    if gp.exists():
        base = json.load(open(gp)).get("walkers", [])
    meas = sorted(glob.glob(str(ep / "measurements" / "*.json.gz")))
    for mp in meas:
        stem = pathlib.Path(mp).name.split(".")[0]
        with gzip.open(mp, "rt") as f:
            m = json.load(f)
        obstacles = m.get("walkers_live") or base
        mat = np.array(m["ego_matrix"])
        ego_xy = mat[:2, 3]
        yaw = float(np.arctan2(mat[1, 0], mat[0, 0]))
        arr = render_route_risk_grid(
            obstacles, ego_xy, yaw, m.get("route") or None,
            TOPDOWN_SIZE, TOPDOWN_HALF, ROUTE_CORRIDOR_HALF_WIDTH,
            resolution=TOPDOWN_RES, horizon_scale=HORIZON_SCALE)
        Image.fromarray(arr).save(hm_dir / f"{stem}.png")
        tp = ep / "topdown" / f"{stem}.jpg"
        if tp.exists():
            td = np.array(Image.open(tp).convert("RGB"))
            Image.fromarray(heatmap_on_topdown(td, arr)).save(tdh_dir / f"{stem}.jpg", quality=90)
    print(f"  {ep.name}: regenerated {len(meas)} frames (gmm_obstacles={len(base)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode", nargs="?", default=None,
                    help="single episode dir (name under --data-dir or full path)")
    ap.add_argument("--all", action="store_true", help="rebake every Town04_* episode")
    ap.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_ROOT)
    ap.add_argument("--reuse-heatmaps", action="store_true",
                    help="build topdown_heatmap/ from existing heatmap/ PNGs (no recompute)")
    args = ap.parse_args()

    if args.all:
        eps = sorted(p for p in glob.glob(str(args.data_dir / "Town04_*"))
                     if os.path.isdir(p))
        print(f"Rebaking {len(eps)} episodes in {args.data_dir}")
        for ep in eps:
            rebake_episode(ep, reuse_heatmaps=args.reuse_heatmaps)
    else:
        if args.episode is None:
            sys.exit("Specify an episode or --all")
        ep = pathlib.Path(args.episode)
        if not ep.exists():
            ep = args.data_dir / args.episode
        if not ep.exists():
            sys.exit(f"Episode not found: {args.episode}")
        rebake_episode(ep, reuse_heatmaps=args.reuse_heatmaps)


if __name__ == "__main__":
    main()
