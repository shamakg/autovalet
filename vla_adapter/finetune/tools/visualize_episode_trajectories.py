"""
Create a video showing the full driven trajectory + vehicle orientation for every episode.

Each video frame = one episode. Shows:
  - The car's world-space path across all saved frames (blue trail)
  - An arrow at each saved position showing heading
  - The destination (red X)
  - The A* route label at the first frame (green line, ego-frame rotated to world)

Usage:
    python visualize_episode_trajectories.py [--data-dir DIR] [--out DIR] [--fps N]
"""

import argparse
import gzip
import io
import json
import os
import pathlib

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_all_frames(ep_dir):
    files = sorted((ep_dir / 'measurements').glob('*.json.gz'))
    frames = []
    for f in files:
        with gzip.open(f, 'rt') as fh:
            frames.append(json.load(fh))
    return frames


def ego_route_to_world(route, ego_x, ego_y, yaw):
    """Convert ego-frame route waypoints back to world coords for overlay."""
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    route = np.array(route)
    world_x = ego_x + route[:, 0] * cos_y - route[:, 1] * sin_y
    world_y = ego_y + route[:, 0] * sin_y + route[:, 1] * cos_y
    return np.stack([world_x, world_y], axis=1)


def render_episode(ep_dir, ax):
    frames = load_all_frames(ep_dir)
    if not frames:
        return False

    # Reconstruct world positions and headings from ego_matrix
    positions = []
    yaws = []
    for d in frames:
        m = np.array(d['ego_matrix'])
        positions.append([m[0, 3], m[1, 3]])
        # heading from rotation matrix column 0 (forward vector)
        yaws.append(np.arctan2(m[1, 0], m[0, 0]))

    positions = np.array(positions)
    yaws = np.array(yaws)

    # Truncate at teleport: any frame-to-frame jump > 5m is a scenario respawn
    if len(positions) > 1:
        jumps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        teleport = np.where(jumps > 5.0)[0]
        if len(teleport) > 0:
            cut = teleport[0] + 1  # keep up to and including the frame before jump
            positions = positions[:cut]
            yaws      = yaws[:cut]
            frames    = frames[:cut]

    # Destination: decode from target_point of first frame + ego pose
    first = frames[0]
    m0 = np.array(first['ego_matrix'])
    ego_x0, ego_y0 = m0[0, 3], m0[1, 3]
    yaw0 = yaws[0]
    tp = np.array(first['target_point'])
    # target_point is in ego frame → convert to world
    cos_y, sin_y = np.cos(yaw0), np.sin(yaw0)
    dest_x = ego_x0 + tp[0] * cos_y - tp[1] * sin_y
    dest_y = ego_y0 + tp[0] * sin_y + tp[1] * cos_y

    ax.clear()

    # Driven path
    ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=1.5, zorder=2)

    # Orientation arrows — subsample to avoid clutter
    step = max(1, len(positions) // 20)
    for i in range(0, len(positions), step):
        ax.annotate('',
            xy=(positions[i, 0] + 1.2 * np.cos(yaws[i]),
                positions[i, 1] + 1.2 * np.sin(yaws[i])),
            xytext=(positions[i, 0], positions[i, 1]),
            arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.5),
            zorder=3)

    # Start and end markers
    ax.plot(positions[0, 0],  positions[0, 1],  'go', markersize=10, zorder=4, label='start')
    ax.plot(positions[-1, 0], positions[-1, 1], 'bs', markersize=8,  zorder=4, label='end')
    ax.plot(dest_x, dest_y, 'rx', markersize=12, markeredgewidth=2.5, zorder=4, label='destination')

    # A* route label at first frame (world frame)
    route_world = ego_route_to_world(first['route'], ego_x0, ego_y0, yaw0)
    ax.plot(route_world[:, 0], route_world[:, 1], 'g--', linewidth=1.2,
            alpha=0.7, zorder=2, label='route label')

    ax.set_aspect('equal')
    ax.set_xlabel('World X (m)')
    ax.set_ylabel('World Y (m)')
    ax.set_title(f'{ep_dir.name}  ({len(frames)} frames,  '
                 f'speed0={first["speed"]:.2f} m/s)')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, linewidth=0.3)
    return True


def make_trajectory_video(data_dir, out_path, fps=6):
    episodes = sorted(data_dir.glob('Town04_*'))
    W, H = 700, 600

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (W, H),
    )

    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)

    for ep_dir in episodes:
        ok = render_episode(ep_dir, ax)
        if not ok:
            continue
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        frame = cv2.resize(frame, (W, H))
        writer.write(frame)

    writer.release()
    plt.close(fig)

    tmp = str(out_path) + '.tmp.mp4'
    os.rename(str(out_path), tmp)
    os.system(f'ffmpeg -y -i "{tmp}" -vcodec libx264 -pix_fmt yuv420p -crf 18 "{out_path}" -loglevel quiet')
    os.remove(tmp)
    print(f'Saved trajectory video to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--out',      default=None)
    parser.add_argument('--fps',      type=int, default=6)
    args = parser.parse_args()

    default_data = pathlib.Path(
        '/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/'
        'vla_adapter/finetune/run_001/data/simlingo/parking_ft/'
        'routes_training/RouteScenario_parking'
    )
    data_dir = pathlib.Path(args.data_dir) if args.data_dir else default_data
    out_path = pathlib.Path(args.out) if args.out else data_dir.parent / 'episode_trajectories.mp4'

    make_trajectory_video(data_dir, out_path, fps=args.fps)
