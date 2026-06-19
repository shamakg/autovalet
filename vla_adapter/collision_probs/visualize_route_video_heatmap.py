#!/usr/bin/env python3
"""visualize_route_video_heatmap.py

Per-episode video for inspecting the collision-risk (danger-zone) heatmap.

Three panels:
  1. RGB front camera.
  2. Top-down camera (raw, ego-attached bird's-eye view).
  3. Top-down + risk overlay: heatmap PNG saved during collection blended onto
     the top-down camera with plasma colormap, alpha 0.45.

The heatmap PNG is written by collect_data_topdown.py at the top-down camera's
exact ground footprint, so no coordinate transform is needed here — just load
and blend.

Usage
-----
    python visualize_route_video_heatmap.py <episode_dir> [--fps N] [--out FILE]
    python visualize_route_video_heatmap.py --all [--data-dir DIR] [--out-dir DIR]
"""
import argparse
import gzip
import json
import pathlib
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

_HERE = pathlib.Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from crossing_gmm import (colorize_soft, route_corridor_mask, clip_route_to_bumper,
                          bake_heatmap_overlay)
from config import ROUTE_CORRIDOR_HALF_WIDTH, TOPDOWN_HEIGHT, TOPDOWN_FOV

import math
TOPDOWN_HALF = TOPDOWN_HEIGHT * math.tan(math.radians(TOPDOWN_FOV / 2.0))

# Muted color palette.
C_BG_FIG = "#131520"
C_BG_BEV = "#1a1c2a"


def _blend_risk(td_img, hm_gray, route_pts, alpha_low=0.25, alpha_high=0.55,
                flip_lr=False, flip_ud=False):
    """Alpha-blend risk heatmap onto a top-down RGB image, clipped to the route corridor."""
    risk = hm_gray.astype(float) / 255.0
    if flip_lr:
        risk = risk[:, ::-1]
    if flip_ud:
        risk = risk[::-1, :]
    h, w = td_img.shape[:2]
    if risk.shape[:2] != (h, w):
        risk = cv2.resize(risk.astype(np.float32), (w, h),
                          interpolation=cv2.INTER_NEAREST)

    if route_pts:
        heatmap_route = clip_route_to_bumper(route_pts)
        inside = route_corridor_mask(heatmap_route, ROUTE_CORRIDOR_HALF_WIDTH, TOPDOWN_HALF, h)
    else:
        inside = np.ones((h, w), dtype=bool)
    risk[~inside] = 0.0

    color_bgr = colorize_soft(risk)
    color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    alpha = (alpha_low + (alpha_high - alpha_low) * np.clip(risk * 3.0, 0, 1))[..., None]
    blended = ((1.0 - alpha) * td_img + alpha * color).astype(np.uint8)
    return np.where(inside[..., None], blended, td_img)


def load_walkers(ep_dir):
    gmm_path = ep_dir / "gmm.json"
    if not gmm_path.exists():
        return []
    with open(gmm_path) as f:
        return json.load(f).get("walkers", [])


def load_episode(ep_dir):
    """Yield (stem, cam_img_or_None) for every frame in the episode.

    Uses rgb/ if present (3-panel mode), otherwise topdown/ (2-panel mode).
    """
    rgb_dir = ep_dir / "rgb"
    td_dir  = ep_dir / "topdown"
    if rgb_dir.is_dir() and any(rgb_dir.iterdir()):
        for img_path in sorted(rgb_dir.glob("*.jpg")):
            yield img_path.stem, np.array(Image.open(img_path))
    elif td_dir.is_dir():
        for img_path in sorted(td_dir.glob("*.jpg")):
            yield img_path.stem, None   # no RGB panel; stem drives frame ordering


def process_episode(ep_dir, fps, out_path, flip_lr=False, flip_ud=False,
                    heatmap_mode='ego_topdown'):
    ep_dir = pathlib.Path(ep_dir)
    meta_path = ep_dir / "episode_meta.json"
    ep_type = "unknown"
    if meta_path.exists():
        with open(meta_path) as fh:
            ep_type = json.load(fh).get("episode_type", "unknown")

    n_walkers = len(load_walkers(ep_dir))
    td_dir  = ep_dir / "topdown"
    hm_dir  = ep_dir / "heatmap"
    has_td  = td_dir.is_dir() and any(td_dir.iterdir())
    has_hm  = hm_dir.is_dir() and any(hm_dir.iterdir())
    print(f"  {ep_dir.name}: walkers={n_walkers} type={ep_type} "
          f"topdown={'yes' if has_td else 'no'} heatmap={'yes' if has_hm else 'no'}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load all frames from disk and pre-blend — pure I/O + cv2, no GMM math.
    print(f"  Loading frames...", end="", flush=True)
    meas_dir = ep_dir / "measurements"

    # Build panel 1 from the PLAIN camera so we can construct the true model input
    # here (via bake_heatmap_overlay) regardless of whether rgb/ was baked in place.
    # rgb_plain/ is the backup rebake.py keeps; fall back to rgb/ (plain on un-baked data).
    plain_dir = ep_dir / "rgb_plain"
    plain_dir = plain_dir if (plain_dir.is_dir() and any(plain_dir.iterdir())) else None

    frames_data = []
    for stem, cam_img in load_episode(ep_dir):
        if plain_dir is not None:
            _p = plain_dir / f"{stem}.jpg"
            if _p.exists():
                cam_img = np.array(Image.open(_p))
        td_path   = td_dir  / f"{stem}.jpg"
        hm_path   = hm_dir  / f"{stem}.png"
        meas_path = meas_dir / f"{stem}.json.gz"
        td_raw  = np.array(Image.open(td_path))[..., :3] \
                  if (has_td and td_path.exists()) else None
        hm_gray = np.array(Image.open(hm_path).convert("L")) \
                  if hm_path.exists() else None
        route_pts = []
        if meas_path.exists():
            with gzip.open(meas_path, "rt") as _f:
                route_pts = json.load(_f).get("route") or []
        if td_raw is not None and hm_gray is not None:
            hm_img  = _blend_risk(td_raw, hm_gray, route_pts,
                                   flip_lr=flip_lr, flip_ud=flip_ud)
            hm_title = "Top-down + risk overlay"
        elif hm_gray is not None:
            hm_img  = hm_gray
            hm_title = "Ego heatmap"
        else:
            hm_img  = None
            hm_title = "No heatmap"
        # Panel 1 = the EXACT model input, constructed with the same
        # bake_heatmap_overlay used by collection + inference (so it's correct
        # whether or not rgb/ on disk has been baked).
        if cam_img is not None and td_raw is not None and hm_gray is not None:
            cam_img = bake_heatmap_overlay(cam_img, td_raw, hm_gray, mode=heatmap_mode)
        frames_data.append((cam_img, td_raw, hm_img, hm_title))
    print(f" done ({len(frames_data)} frames)")

    has_rgb = any(c is not None for c, *_ in frames_data)
    if has_rgb:
        # rgb/ is what collect_data_topdown.py actually saves to disk, and since
        # baking happens at collection time (see bake_heatmap_overlay in
        # crossing_gmm.py), this panel IS the exact model input -- no further
        # processing happens in the dataloader. Give it extra width since a
        # baked frame is a 2:1 wide [camera | topdown+heatmap] image.
        fig = plt.figure(figsize=(24, 6), facecolor=C_BG_FIG)
        gs  = fig.add_gridspec(1, 3, width_ratios=[8, 3, 3],
                               left=0.01, right=0.99, top=0.93, bottom=0.06, wspace=0.06)
        ax_cam, ax_td, ax_hm = (fig.add_subplot(gs[0, k]) for k in range(3))
    else:
        # Two panels: Top-down raw | Top-down + overlay
        fig = plt.figure(figsize=(12, 6), facecolor=C_BG_FIG)
        gs  = fig.add_gridspec(1, 2, width_ratios=[1, 1],
                               left=0.01, right=0.99, top=0.93, bottom=0.06, wspace=0.06)
        ax_cam = None
        ax_td, ax_hm = (fig.add_subplot(gs[0, k]) for k in range(2))

    writer = None
    for cam_img, td_raw, hm_img, hm_title in frames_data:

        if ax_cam is not None:
            ax_cam.clear(); ax_cam.axis("off")
            if cam_img is not None:
                ax_cam.imshow(cam_img)
            ax_cam.set_title(f"Model input ({heatmap_mode}) -- exactly what the model sees",
                             fontsize=9, color="white", pad=3)

        ax_td.clear(); ax_td.axis("off")
        if td_raw is not None:
            ax_td.imshow(td_raw)
            ax_td.set_title("Top-down camera", fontsize=9, color="white", pad=3)
        else:
            ax_td.set_facecolor(C_BG_BEV)
            ax_td.set_title("Top-down (none)", fontsize=9, color="#666688", pad=3)

        ax_hm.clear(); ax_hm.axis("off")
        if hm_img is not None:
            kw = {"cmap": "plasma", "vmin": 0, "vmax": 255} if hm_img.ndim == 2 else {}
            ax_hm.imshow(hm_img, **kw)
        ax_hm.set_title(hm_title, fontsize=9, color="white", pad=3)

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        frame = buf.reshape(h, w, 4)[..., :3]
        if writer is None:
            writer = cv2.VideoWriter(str(out_path),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    plt.close(fig)
    if writer is not None:
        writer.release()
        print(f"  -> {out_path}")
    else:
        print(f"  -> no frames in {ep_dir.name}")


def main():
    default_root = (
        _HERE / "run_topdown_001" / "data" / "simlingo" / "parking_ft"
        / "routes_training" / "RouteScenario_parking"
    )
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode", nargs="?", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--data-dir", type=pathlib.Path, default=default_root)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--out-dir", type=pathlib.Path, default=None)
    ap.add_argument("--flip-lr", action="store_true",
                    help="mirror the heatmap horizontally (if camera axis differs)")
    ap.add_argument("--flip-ud", action="store_true",
                    help="mirror the heatmap vertically")
    ap.add_argument("--heatmap-mode", default="ego_topdown",
                    choices=["ego_topdown", "topdown_only"],
                    help="how panel 1 (the model input) is composed")
    args = ap.parse_args()

    if args.all:
        if not args.data_dir.exists():
            sys.exit(f"Data dir not found: {args.data_dir}")
        out_dir  = args.out_dir or args.data_dir.parent / "heatmap_videos"
        episodes = sorted(args.data_dir.glob("Town04_*"))
        print(f"Processing {len(episodes)} episodes -> {out_dir}")
        for ep in episodes:
            process_episode(ep, args.fps, out_dir / f"{ep.name}_heatmap.mp4",
                            flip_lr=args.flip_lr, flip_ud=args.flip_ud,
                            heatmap_mode=args.heatmap_mode)
    else:
        if args.episode is None:
            episodes = sorted(default_root.glob("Town04_*"))
            if not episodes:
                sys.exit(f"No episodes found in {default_root}")
            ep_dir = episodes[0]
        else:
            ep_dir = pathlib.Path(args.episode)
            if not ep_dir.exists():
                ep_dir = default_root / args.episode
            if not ep_dir.exists():
                sys.exit(f"Episode directory not found: {args.episode}")
        out_path = args.out or ep_dir.parent / f"{ep_dir.name}_heatmap.mp4"
        process_episode(ep_dir, args.fps, out_path,
                        flip_lr=args.flip_lr, flip_ud=args.flip_ud,
                        heatmap_mode=args.heatmap_mode)


if __name__ == "__main__":
    main()
