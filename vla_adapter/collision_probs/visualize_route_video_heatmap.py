#!/usr/bin/env python3
"""visualize_route_video_heatmap.py

Per-episode video showing every view saved by collect_data_topdown.py.

Three panels:
  1. rgb           — plain front camera (view 1 fed to the model).
  2. rgb_augmented — laterally-shifted augmented front camera.
  3. topdown_heatmap — BEV camera + colorized risk overlay, already baked at
     collection time (view 2 fed to the model, use_topdown=true).

All three are saved pre-baked to disk by collect_data_topdown.py, so this
script just loads and displays them side by side -- no blending or coordinate
transforms happen here.

Usage
-----
    python visualize_route_video_heatmap.py <episode_dir> [--fps N] [--out FILE]
    python visualize_route_video_heatmap.py --all [--data-dir DIR] [--out-dir DIR]
"""
import argparse
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

# Muted color palette.
C_BG_FIG = "#131520"
C_BG_BEV = "#1a1c2a"


def load_episode(ep_dir):
    """Yield frame stems, driven by rgb/ (always present per collect_data_topdown.py)."""
    rgb_dir = ep_dir / "rgb"
    if rgb_dir.is_dir():
        for img_path in sorted(rgb_dir.glob("*.jpg")):
            yield img_path.stem


def process_episode(ep_dir, fps, out_path):
    ep_dir = pathlib.Path(ep_dir)
    meta_path = ep_dir / "episode_meta.json"
    ep_type = "unknown"
    if meta_path.exists():
        with open(meta_path) as fh:
            ep_type = json.load(fh).get("episode_type", "unknown")

    rgb_dir = ep_dir / "rgb"
    aug_dir = ep_dir / "rgb_augmented"
    hm_dir  = ep_dir / "topdown_heatmap"
    has_aug = aug_dir.is_dir() and any(aug_dir.iterdir())
    has_hm  = hm_dir.is_dir() and any(hm_dir.iterdir())
    print(f"  {ep_dir.name}: type={ep_type} "
          f"rgb_augmented={'yes' if has_aug else 'no'} "
          f"topdown_heatmap={'yes' if has_hm else 'no'}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Loading frames...", end="", flush=True)
    frames_data = []
    for stem in load_episode(ep_dir):
        rgb_img = np.array(Image.open(rgb_dir / f"{stem}.jpg"))
        aug_path = aug_dir / f"{stem}.jpg"
        hm_path  = hm_dir  / f"{stem}.jpg"
        aug_img = np.array(Image.open(aug_path)) if aug_path.exists() else None
        hm_img  = np.array(Image.open(hm_path))  if hm_path.exists()  else None
        frames_data.append((rgb_img, aug_img, hm_img))
    print(f" done ({len(frames_data)} frames)")

    fig = plt.figure(figsize=(18, 6), facecolor=C_BG_FIG)
    gs  = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1],
                           left=0.01, right=0.99, top=0.93, bottom=0.06, wspace=0.06)
    ax_rgb, ax_aug, ax_hm = (fig.add_subplot(gs[0, k]) for k in range(3))

    writer = None
    for rgb_img, aug_img, hm_img in frames_data:
        ax_rgb.clear(); ax_rgb.axis("off")
        ax_rgb.imshow(rgb_img)
        ax_rgb.set_title("rgb -- front camera (view 1)", fontsize=9, color="white", pad=3)

        ax_aug.clear(); ax_aug.axis("off")
        if aug_img is not None:
            ax_aug.imshow(aug_img)
            ax_aug.set_title("rgb_augmented -- lateral shift", fontsize=9, color="white", pad=3)
        else:
            ax_aug.set_facecolor(C_BG_BEV)
            ax_aug.set_title("rgb_augmented (none)", fontsize=9, color="#666688", pad=3)

        ax_hm.clear(); ax_hm.axis("off")
        if hm_img is not None:
            ax_hm.imshow(hm_img)
            ax_hm.set_title("topdown_heatmap -- BEV + risk (view 2)", fontsize=9, color="white", pad=3)
        else:
            ax_hm.set_facecolor(C_BG_BEV)
            ax_hm.set_title("topdown_heatmap (none)", fontsize=9, color="#666688", pad=3)

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
    args = ap.parse_args()

    if args.all:
        if not args.data_dir.exists():
            sys.exit(f"Data dir not found: {args.data_dir}")
        out_dir  = args.out_dir or args.data_dir.parent / "heatmap_videos"
        episodes = sorted(args.data_dir.glob("Town04_*"))
        print(f"Processing {len(episodes)} episodes -> {out_dir}")
        for ep in episodes:
            process_episode(ep, args.fps, out_dir / f"{ep.name}_heatmap.mp4")
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
        process_episode(ep_dir, args.fps, out_path)


if __name__ == "__main__":
    main()
