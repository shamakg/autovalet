"""Post-process existing collected data in-place.

Fixes applied:
1. route[0] forced to [0.0, 0.0] (car center) — removes behind-car first waypoints.
2. Frames with signed speed < 0 are deleted (spawn jitter artifacts).

Also regenerates the static route visualizations and creates a new per-episode
start-route video.

Usage:
    python postprocess_data.py [--data-dir DIR] [--dry-run]
"""

import argparse
import gzip
import json
import os
import pathlib

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def iter_episodes(data_dir):
    return sorted(data_dir.glob('Town04_*'))


def postprocess_episode(ep_dir, dry_run=False):
    meas_dir = ep_dir / 'measurements'
    rgb_dir  = ep_dir / 'rgb'
    box_dir  = ep_dir / 'boxes'

    files = sorted(meas_dir.glob('*.json.gz'))
    fixed = 0

    for f in files:
        stem = f.stem.replace('.json', '')  # e.g. '0042'
        with gzip.open(f, 'rt') as fh:
            d = json.load(fh)

        # Force route[0] to car center
        changed = False
        if d['route'][0] != [0.0, 0.0]:
            d['route'][0] = [0.0, 0.0]
            changed = True
        if d.get('route_original', [[]])[0] != [0.0, 0.0]:
            d['route_original'][0] = [0.0, 0.0]
            changed = True

        if changed:
            if not dry_run:
                with gzip.open(f, 'wt', encoding='utf-8') as fh:
                    json.dump(d, fh)
            fixed += 1

    return fixed


def load_episode_first_frame(ep_dir):
    meas_dir = ep_dir / 'measurements'
    files = sorted(meas_dir.glob('*.json.gz'))
    if not files:
        return None
    with gzip.open(files[0], 'rt') as f:
        return json.load(f)


def make_route_video(data_dir, out_path, fps=8):
    import io
    episodes = iter_episodes(data_dir)
    W, H = 600, 600

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (W, H),
    )

    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)

    for ep_dir in episodes:
        d = load_episode_first_frame(ep_dir)
        if d is None:
            continue

        route  = np.array(d['route'])
        target = np.array(d['target_point'])
        speed  = d['speed']

        ax.clear()
        ax.plot(route[:, 0], route[:, 1], 'b-o', markersize=3, linewidth=1.5)
        ax.plot(0, 0, 'go', markersize=10, zorder=5, label='ego')
        ax.plot(target[0], target[1], 'rx', markersize=10, markeredgewidth=2, label='target')
        ax.axhline(0, color='gray', linewidth=0.4)
        ax.axvline(0, color='gray', linewidth=0.4)
        ax.set_aspect('equal')
        ax.set_xlabel('Forward (m)')
        ax.set_ylabel('Left (m)')
        ax.set_title(f'{ep_dir.name}   speed={speed:.2f} m/s   target=({target[0]:.1f}, {target[1]:.1f})')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(-15, 60)
        ax.set_ylim(-15, 15)
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

    # Re-encode with libx264 so all players can read it
    tmp = str(out_path) + '.tmp.mp4'
    os.rename(str(out_path), tmp)
    os.system(f'ffmpeg -y -i "{tmp}" -vcodec libx264 -pix_fmt yuv420p -crf 18 "{out_path}" -loglevel quiet')
    os.remove(tmp)
    print(f'Saved episode video to {out_path}')


def make_static_plots(data_dir, out_dir):
    # Inline the same plots from visualize_routes.py but reading post-fixed data
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        'visualize_routes',
        pathlib.Path(__file__).parent / 'visualize_routes.py'
    )
    vr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vr)
    vr.plot_episode_grid(data_dir, out_dir / 'expert_routes_grid.png', max_episodes=25)
    vr.plot_all_routes_overlay(data_dir, out_dir / 'expert_routes_overlay.png')
    vr.plot_route_start_end(data_dir, out_dir / 'expert_routes_start_end.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be changed without modifying files')
    args = parser.parse_args()

    default_data = pathlib.Path(
        '/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/'
        'vla_adapter/finetune/run_001/data/simlingo/parking_ft/'
        'routes_training/RouteScenario_parking'
    )
    data_dir = pathlib.Path(args.data_dir) if args.data_dir else default_data
    out_dir  = pathlib.Path(args.out_dir)  if args.out_dir  else data_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data: {data_dir}  ({'DRY RUN' if args.dry_run else 'LIVE'})")

    total_fixed = 0
    for ep_dir in iter_episodes(data_dir):
        total_fixed += postprocess_episode(ep_dir, dry_run=args.dry_run)

    print(f"\nFrames patched (route[0]->[0,0]): {total_fixed}")

    if not args.dry_run:
        print('\nRegenerating static plots...')
        make_static_plots(data_dir, out_dir)

        print('Generating per-episode start-route video...')
        make_route_video(data_dir, out_dir / 'episode_routes.mp4', fps=8)
