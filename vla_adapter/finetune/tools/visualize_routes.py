"""Visualize expert trajectories from collected training data.

Produces two plots:
1. Grid of ego-frame routes sampled across episodes (one subplot per episode,
   multiple timesteps overlaid). Shows whether routes extend forward or behind.
2. All routes from all episodes overlaid in a single ego-frame plot, colored
   by timestep progression. Reveals systematic biases.

Usage:
    python visualize_routes.py [--data-dir DIR] [--out-dir DIR] [--max-episodes N]
"""

import argparse
import gzip
import json
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_episode_routes(episode_dir):
    meas_dir = episode_dir / 'measurements'
    if not meas_dir.exists():
        return []
    files = sorted(meas_dir.glob('*.json.gz'))
    routes = []
    for f in files:
        with gzip.open(f, 'rt') as fh:
            d = json.load(fh)
        route = np.array(d['route'])
        target = np.array(d['target_point'])
        speed = d.get('speed', 0.0)
        routes.append({'route': route, 'target': target, 'speed': speed, 'frame': f.stem})
    return routes


def plot_episode_grid(data_dir, out_path, max_episodes=25):
    episodes = sorted(data_dir.glob('Town04_*'))
    if not episodes:
        print(f"No episodes found in {data_dir}")
        return

    step = max(1, len(episodes) // max_episodes)
    sampled = episodes[::step][:max_episodes]
    cols = 5
    rows = (len(sampled) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for idx, ep_dir in enumerate(sampled):
        ax = axes[idx]
        routes = load_episode_routes(ep_dir)
        if not routes:
            ax.set_title(ep_dir.name + ' (empty)')
            continue

        n = len(routes)
        sample_indices = np.linspace(0, n - 1, min(n, 12), dtype=int)
        for i in sample_indices:
            r = routes[i]
            alpha = 0.3 + 0.7 * (i / max(n - 1, 1))
            ax.plot(r['route'][:, 0], r['route'][:, 1], 'b-', alpha=alpha, linewidth=0.8)
            ax.plot(r['target'][0], r['target'][1], 'rx', markersize=4, alpha=alpha)

        ax.plot(0, 0, 'go', markersize=6)
        ax.axhline(0, color='gray', linewidth=0.3)
        ax.axvline(0, color='gray', linewidth=0.3)
        ax.set_aspect('equal')
        ax.set_title(ep_dir.name, fontsize=8)
        ax.tick_params(labelsize=6)

    for idx in range(len(sampled), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Expert routes in ego frame (blue=route, red=target, green=ego)\n'
                 'X = forward, Y = left', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved grid plot to {out_path}')


def plot_all_routes_overlay(data_dir, out_path, max_episodes=50):
    episodes = sorted(data_dir.glob('Town04_*'))
    step = max(1, len(episodes) // max_episodes)
    sampled = episodes[::step][:max_episodes]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: all routes colored by frame progress (early=light, late=dark)
    ax = axes[0]
    for ep_dir in sampled:
        routes = load_episode_routes(ep_dir)
        n = len(routes)
        for i, r in enumerate(routes):
            frac = i / max(n - 1, 1)
            color = plt.cm.viridis(frac)
            ax.plot(r['route'][:, 0], r['route'][:, 1], color=color, alpha=0.15, linewidth=0.5)
    ax.plot(0, 0, 'ro', markersize=8, zorder=5)
    ax.set_aspect('equal')
    ax.set_xlabel('Forward (m)')
    ax.set_ylabel('Left (m)')
    ax.set_title('All ego-frame routes (color = time progression)')
    ax.axhline(0, color='gray', linewidth=0.3)
    ax.axvline(0, color='gray', linewidth=0.3)

    # Right: histogram of route[0] (first waypoint) distance from ego
    first_wp_dists = []
    first_wp_x = []
    for ep_dir in sampled:
        routes = load_episode_routes(ep_dir)
        for r in routes:
            wp0 = r['route'][0]
            first_wp_dists.append(np.linalg.norm(wp0))
            first_wp_x.append(wp0[0])

    ax2 = axes[1]
    ax2.hist(first_wp_x, bins=80, edgecolor='black', linewidth=0.3)
    ax2.set_xlabel('route[0].x  (forward component, m)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'First waypoint forward offset\n(negative = behind car)')
    ax2.axvline(0, color='red', linewidth=1, linestyle='--', label='ego position')
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved overlay plot to {out_path}')


def plot_route_start_end(data_dir, out_path, max_episodes=80):
    """Scatter of route[0] and route[-1] across all frames — shows if start/end are sane."""
    episodes = sorted(data_dir.glob('Town04_*'))
    step = max(1, len(episodes) // max_episodes)
    sampled = episodes[::step][:max_episodes]

    starts_x, starts_y = [], []
    ends_x, ends_y = [], []
    for ep_dir in sampled:
        routes = load_episode_routes(ep_dir)
        for r in routes:
            starts_x.append(r['route'][0][0])
            starts_y.append(r['route'][0][1])
            ends_x.append(r['route'][-1][0])
            ends_y.append(r['route'][-1][1])

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(starts_x, starts_y, s=1, alpha=0.2, c='blue', label='route[0] (start)')
    ax.scatter(ends_x, ends_y, s=1, alpha=0.2, c='red', label='route[-1] (end)')
    ax.plot(0, 0, 'k*', markersize=12, zorder=5, label='ego')
    ax.set_aspect('equal')
    ax.set_xlabel('Forward (m)')
    ax.set_ylabel('Left (m)')
    ax.set_title('Route start/end points in ego frame')
    ax.legend()
    ax.axhline(0, color='gray', linewidth=0.3)
    ax.axvline(0, color='gray', linewidth=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved start/end scatter to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=None,
                        help='Path to RouteScenario_parking directory containing Town04_* episodes')
    parser.add_argument('--out-dir', default=None, help='Where to save plots')
    parser.add_argument('--max-episodes', type=int, default=25, help='Max episodes for grid plot')
    args = parser.parse_args()

    default_data = pathlib.Path(
        '/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/'
        'vla_adapter/finetune/run_001/data/simlingo/parking_ft/'
        'routes_training/RouteScenario_parking'
    )
    data_dir = pathlib.Path(args.data_dir) if args.data_dir else default_data
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else data_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data: {data_dir}")
    print(f"Output: {out_dir}")

    plot_episode_grid(data_dir, out_dir / 'expert_routes_grid.png', max_episodes=args.max_episodes)
    plot_all_routes_overlay(data_dir, out_dir / 'expert_routes_overlay.png')
    plot_route_start_end(data_dir, out_dir / 'expert_routes_start_end.png')
