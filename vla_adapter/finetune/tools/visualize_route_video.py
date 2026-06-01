#!/usr/bin/env python3
"""visualize_route_video.py

Side-by-side video per episode:
  Left  — RGB camera frame
  Right — top-down parking-lot BEV showing destination spot, ego car
          (position + heading), and the 20 route waypoints stored in
          each measurement JSON.

Usage
-----
    # single episode
    python visualize_route_video.py <episode_dir> [--fps N] [--out FILE]

    # all episodes under the default data root
    python visualize_route_video.py --all [--fps N] [--out-dir DIR]

    # explicit data root with all episodes
    python visualize_route_video.py --all --data-dir /path/to/RouteScenario_parking

Examples
--------
    python visualize_route_video.py run_001/data/.../Town04_0080 --fps 10
    python visualize_route_video.py --all --fps 8 --out-dir /tmp/route_vids
"""

import argparse
import gzip
import json
import pathlib
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Parking spot world coordinates (mirrors parking_position.py without carla)
# 4 rows × 16 spots, 0-indexed, matching PARKING_SPOTS list in parking_position.py
# ---------------------------------------------------------------------------
_Y_VALS = [-235.73, -232.73, -229.53, -226.43, -223.43, -220.23,
           -217.23, -214.03, -210.73, -207.30, -204.23, -201.03,
           -198.03, -194.90, -191.53, -188.20]

PARKING_SPOTS = np.array(
    [[298.5, y] for y in _Y_VALS] +  # row 1, indices  0-15
    [[290.9, y] for y in _Y_VALS] +  # row 2, indices 16-31
    [[280.0, y] for y in _Y_VALS] +  # row 3, indices 32-47
    [[272.5, y] for y in _Y_VALS],   # row 4, indices 48-63
    dtype=float,
)

# Lane centrelines for context lines
LANE_X = [303.95, 285.45, 267.05]

# BEV viewport (world coords) with a little margin
BEV_XLIM = (263.0, 312.0)
BEV_YLIM = (-248.0, -183.0)

# Approximate car half-dimensions in metres (ego frame: x=forward, y=left)
CAR_HALF_L = 2.3
CAR_HALF_W = 1.0

# Spot rectangle size for destination highlight (world frame, x=row depth, y=slot width)
SPOT_HALF_X = 2.8
SPOT_HALF_Y = 1.4


def ego_to_world(points_ego: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Transform (N,2) ego-frame points to world frame using 4×4 ego_matrix."""
    R = M[:2, :2]
    t = M[:2, 3]
    return (R @ points_ego.T).T + t


def car_polygon_world(M: np.ndarray):
    """Return (4,2) world-frame corners of the ego car rectangle."""
    corners_ego = np.array([
        [ CAR_HALF_L,  CAR_HALF_W],
        [ CAR_HALF_L, -CAR_HALF_W],
        [-CAR_HALF_L, -CAR_HALF_W],
        [-CAR_HALF_L,  CAR_HALF_W],
    ])
    return ego_to_world(corners_ego, M)


def draw_bev(ax, dest_idx: int, M: np.ndarray, route_ego: np.ndarray,
             target_world: np.ndarray, frame_num: int, speed: float,
             episode_type: str):
    ax.clear()
    ax.set_facecolor("#1a1a2e")
    ax.set_xlim(*BEV_XLIM)
    ax.set_ylim(*BEV_YLIM)
    ax.invert_xaxis()
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    ax.set_xlabel("World X (m)", fontsize=8)
    ax.set_ylabel("World Y (m)", fontsize=8)

    # Lane guide lines
    for lx in LANE_X:
        ax.axvline(lx, color="#444466", linewidth=0.8, linestyle="--")

    # All parking spots
    ax.scatter(PARKING_SPOTS[:, 0], PARKING_SPOTS[:, 1],
               s=18, color="#888899", zorder=2, linewidths=0)

    # Destination spot (highlighted rectangle + centre dot)
    dest_xy = PARKING_SPOTS[dest_idx]
    dest_rect = mpatches.FancyBboxPatch(
        (dest_xy[0] - SPOT_HALF_X, dest_xy[1] - SPOT_HALF_Y),
        2 * SPOT_HALF_X, 2 * SPOT_HALF_Y,
        boxstyle="square,pad=0",
        linewidth=1.5, edgecolor="#ff9900", facecolor="#ff990033", zorder=3,
    )
    ax.add_patch(dest_rect)
    ax.scatter([dest_xy[0]], [dest_xy[1]], s=60, color="#ff9900",
               zorder=4, marker="*")

    # Route waypoints → world frame
    route_world = ego_to_world(route_ego, M)
    ax.plot(route_world[:, 0], route_world[:, 1],
            color="#4fc3f7", linewidth=1.2, zorder=5)
    ax.scatter(route_world[:, 0], route_world[:, 1],
               s=14, color="#4fc3f7", zorder=6, linewidths=0)
    # Mark first and last explicitly
    ax.scatter([route_world[0, 0]], [route_world[0, 1]],
               s=40, color="#00e5ff", zorder=7, marker="o")
    ax.scatter([route_world[-1, 0]], [route_world[-1, 1]],
               s=50, color="#00bcd4", zorder=7, marker="D")

    # Target point (star)
    ax.scatter([target_world[0]], [target_world[1]],
               s=120, color="#f44336", zorder=8, marker="*")

    # Ego car rectangle
    ego_pos = M[:2, 3]
    corners = car_polygon_world(M)
    poly = plt.Polygon(corners, closed=True,
                       facecolor="#76ff03", edgecolor="#ccff00",
                       linewidth=1.2, zorder=9, alpha=0.85)
    ax.add_patch(poly)

    # Forward arrow
    fwd = M[:2, 0]          # first column of R = local x-axis in world
    ax.annotate("", xy=ego_pos + fwd * 3.5, xytext=ego_pos,
                arrowprops=dict(arrowstyle="-|>", color="#ccff00",
                                lw=1.5, mutation_scale=12),
                zorder=10)

    # Legend / info text
    info = (f"Frame {frame_num:04d}  |  speed {speed:+.1f} m/s\n"
            f"Type: {episode_type}  |  dest spot #{dest_idx}")
    ax.text(0.02, 0.98, info, transform=ax.transAxes,
            fontsize=7.5, verticalalignment="top",
            color="white", family="monospace",
            bbox=dict(facecolor="#00000088", edgecolor="none", pad=3))

    # Compact legend
    handles = [
        mpatches.Patch(color="#ff9900", label="destination"),
        mpatches.Patch(color="#4fc3f7", label="route wps"),
        mpatches.Patch(color="#f44336", label="target point"),
        mpatches.Patch(color="#76ff03", label="ego car"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7,
              facecolor="#111122", edgecolor="#444466", labelcolor="white")




def load_episode(ep_dir: pathlib.Path):
    """Yield (frame_idx, rgb_array, aug_rgb_or_None, measurement_dict) per frame."""
    meas_dir    = ep_dir / "measurements"
    rgb_dir     = ep_dir / "rgb"
    aug_rgb_dir = ep_dir / "rgb_augmented"
    for meas_path in sorted(meas_dir.glob("*.json.gz")):
        stem = meas_path.name.replace(".json.gz", "")
        img_path = rgb_dir / f"{stem}.jpg"
        if not img_path.exists():
            continue
        with gzip.open(meas_path, "rt") as fh:
            meas = json.load(fh)
        cam = np.array(Image.open(img_path))
        aug_path = aug_rgb_dir / f"{stem}.jpg"
        aug_cam = np.array(Image.open(aug_path)) if aug_path.exists() else None
        yield int(stem), cam, aug_cam, meas


def process_episode(ep_dir: pathlib.Path, fps: int, out_path: pathlib.Path):
    meta_path = ep_dir / "episode_meta.json"
    if not meta_path.exists():
        print(f"  Skipping {ep_dir.name}: no episode_meta.json")
        return

    with open(meta_path) as fh:
        meta = json.load(fh)
    dest_idx = meta["destination"]
    ep_type  = meta.get("episode_type", "unknown")
    aug_t    = meta.get("aug_translation", 0.0)  # metres; 0 if old episode
    aug_r    = meta.get("aug_rotation_deg", 0.0) # degrees

    has_aug = (ep_dir / "rgb_augmented").exists() and any(
        (ep_dir / "rgb_augmented").iterdir()
    )
    print(f"  {ep_dir.name}  dest={dest_idx}  type={ep_type}  aug={'yes' if has_aug else 'no'}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Layout: 3 columns when augmented camera data exists, 2 otherwise.
    # Widths: main-cam 5, BEV 5, aug-cam 2.5 (half-height via aspect ratio)
    if has_aug:
        fig = plt.figure(figsize=(16, 6), facecolor="#111122")
        gs  = fig.add_gridspec(2, 3, width_ratios=[4, 4, 2],
                               left=0.03, right=0.98, top=0.93, bottom=0.07,
                               wspace=0.06, hspace=0.3)
        ax_cam = fig.add_subplot(gs[:, 0])   # main cam — full height
        ax_bev = fig.add_subplot(gs[:, 1])   # BEV — full height
        ax_aug = fig.add_subplot(gs[0, 2])   # aug cam — top half of right column
        ax_aug_info = fig.add_subplot(gs[1, 2])  # aug metadata — bottom half
        ax_aug.set_facecolor("black")
        ax_aug_info.set_facecolor("#111122")
        ax_aug_info.axis("off")
    else:
        fig = plt.figure(figsize=(16, 6), facecolor="#111122")
        fig.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.07,
                            wspace=0.06)
        ax_cam = fig.add_subplot(1, 2, 1)
        ax_bev = fig.add_subplot(1, 2, 2)
        ax_aug = None
        ax_aug_info = None

    ax_cam.set_facecolor("black")

    writer = None

    for frame_num, cam_img, aug_img, meas in load_episode(ep_dir):
        M            = np.array(meas["ego_matrix"])
        route_ego    = np.array(meas["route"])
        target_ego   = np.array(meas["target_point"])
        target_world = ego_to_world(target_ego[None], M)[0]
        speed        = float(meas.get("speed", 0.0))
        aug_t_meas = float(meas.get("augmentation_translation", aug_t))
        aug_r_meas = float(meas.get("augmentation_rotation", aug_r))  # stored in degrees

        # Main camera + BEV
        ax_cam.clear()
        ax_cam.imshow(cam_img)
        ax_cam.axis("off")
        ax_cam.set_title("RGB camera", fontsize=9, color="white", pad=3)

        draw_bev(ax_bev, dest_idx, M, route_ego, target_world,
                 frame_num, speed, ep_type)
        ax_bev.set_title("Parking lot BEV (world frame)", fontsize=9,
                         color="white", pad=3)

        # Augmented camera panel
        if ax_aug is not None:
            ax_aug.clear()
            if aug_img is not None:
                ax_aug.imshow(aug_img)
            else:
                ax_aug.text(0.5, 0.5, "no aug frame", transform=ax_aug.transAxes,
                            ha="center", va="center", color="#888888", fontsize=8)
            ax_aug.axis("off")
            ax_aug.set_title("Augmented cam", fontsize=8, color="white", pad=2)

            ax_aug_info.clear()
            ax_aug_info.axis("off")
            info = (f"Δy = {aug_t_meas:+.2f} m\n"
                    f"Δyaw = {aug_r_meas:+.1f}°")
            ax_aug_info.text(0.5, 0.5, info, transform=ax_aug_info.transAxes,
                             ha="center", va="center", color="white",
                             fontsize=8, family="monospace",
                             bbox=dict(facecolor="#00000088", edgecolor="none", pad=4))

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        frame = buf.reshape(h, w, 4)[..., :3]

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    plt.close(fig)
    if writer is not None:
        writer.release()
        print(f"  -> {out_path}")
    else:
        print(f"  -> no frames found in {ep_dir.name}")


def main():
    default_root = pathlib.Path(
        "/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet"
        "/vla_adapter/finetune/run_001/data/simlingo/parking_ft"
        "/routes_training/RouteScenario_parking"
    )

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episode", nargs="?", default=None,
                        help="Path to a single Town04_XXXX episode directory")
    parser.add_argument("--all", action="store_true",
                        help="Process every episode under --data-dir")
    parser.add_argument("--data-dir", type=pathlib.Path, default=default_root,
                        help="Root RouteScenario_parking directory (for --all)")
    parser.add_argument("--fps", type=int, default=10,
                        help="Output video frame rate (default: 10)")
    parser.add_argument("--out", type=pathlib.Path, default=None,
                        help="Output file for single-episode mode")
    parser.add_argument("--out-dir", type=pathlib.Path, default=None,
                        help="Output directory for --all mode")
    args = parser.parse_args()

    if args.all:
        data_dir = args.data_dir
        if not data_dir.exists():
            sys.exit(f"Data dir not found: {data_dir}")
        out_dir = args.out_dir or data_dir.parent / "route_videos"
        episodes = sorted(data_dir.glob("Town04_*"))
        print(f"Processing {len(episodes)} episodes -> {out_dir}")
        for ep in episodes:
            out_path = out_dir / f"{ep.name}_route.mp4"
            process_episode(ep, args.fps, out_path)
    else:
        if args.episode is None:
            # Default: first episode in the data root
            episodes = sorted(default_root.glob("Town04_*"))
            if not episodes:
                sys.exit(f"No episodes found in {default_root}")
            ep_dir = episodes[0]
            print(f"No episode given; using {ep_dir}")
        else:
            ep_dir = pathlib.Path(args.episode)
            if not ep_dir.exists():
                # Try relative to data root
                ep_dir = default_root / args.episode
            if not ep_dir.exists():
                sys.exit(f"Episode directory not found: {args.episode}")

        out_path = args.out or ep_dir.parent / f"{ep_dir.name}_route.mp4"
        print(f"Episode: {ep_dir}")
        process_episode(ep_dir, args.fps, out_path)


if __name__ == "__main__":
    main()
