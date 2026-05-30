#!/usr/bin/env python3
"""
Randomly sample episodes, encode RGB frames into a video, print the path,
wait for you to watch it, then delete it and move on.

Usage:
    python validate_episodes.py <run_dir> [--n N] [--fps FPS]

run_dir: e.g. finetune/run_001_backup/data/simlingo/parking_ft/routes_training/RouteScenario_parking
"""

import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import sys


def find_episodes(run_dir):
    rgb_dirs = glob.glob(os.path.join(run_dir, "**/rgb"), recursive=True)
    return sorted(os.path.dirname(d) for d in rgb_dirs if os.path.isdir(d))


def load_meta(episode_dir):
    path = os.path.join(episode_dir, "episode_meta.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def make_video(rgb_dir, fps, output_path):
    frames = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")))
    if not frames:
        return False
    list_path = output_path + ".txt"
    with open(list_path, "w") as f:
        for frame in frames:
            f.write(f"file '{os.path.abspath(frame)}'\n")
            f.write(f"duration {1.0/fps:.6f}\n")
    result = subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         output_path],
        capture_output=True,
    )
    os.remove(list_path)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("ERROR: ffmpeg not found.")

    run_dir = os.path.abspath(args.run_dir)
    episodes = find_episodes(run_dir)
    if not episodes:
        sys.exit(f"No episodes found under {run_dir}")

    n = min(args.n, len(episodes))
    sample = random.sample(episodes, n)
    print(f"Found {len(episodes)} episodes. Sampling {n}.\n")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "review")
    os.makedirs(out_dir, exist_ok=True)
    video_path = os.path.join(out_dir, "current_episode.mp4")

    for i, episode_dir in enumerate(sample, 1):
        rgb_dir = os.path.join(episode_dir, "rgb")
        frames = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")))
        meta = load_meta(episode_dir)
        name = os.path.basename(episode_dir)

        iou = meta.get("iou")
        iou_str = f"{iou:.4f}" if isinstance(iou, float) else "N/A"
        print(f"{'='*55}")
        print(f"  Episode {i}/{n}: {name}")
        print(f"  Type:        {meta.get('episode_type', 'N/A')}")
        print(f"  Destination: {meta.get('destination', 'N/A')}")
        print(f"  IoU:         {iou_str}")
        print(f"  Collisions:  vehicle={meta.get('vehicle_collisions','N/A')}  walker={meta.get('walker_collisions','N/A')}")
        print(f"  Frames:      {len(frames)}")
        print(f"{'='*55}")

        print("Building MP4... ", end="", flush=True)
        if not make_video(rgb_dir, args.fps, video_path):
            print("ffmpeg failed — skipping.")
            continue
        print(f"done\n")
        print(f"  {video_path}\n")

        try:
            input("Press Enter when done to delete & continue (Ctrl+C to quit)... ")
        except KeyboardInterrupt:
            print("\nStopped.")
            break

        os.remove(video_path)
        print("Deleted.\n")

    print("Done.")


if __name__ == "__main__":
    main()
