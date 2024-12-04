"""
plotting.py — plot IOU and collision rate for static and dynamic scenarios separately.

Run:  python plotting.py
"""

import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

OUT_STATIC  = 'scenario_comparison_static.png'
OUT_DYNAMIC = 'scenario_comparison_dynamic.png'

_LABEL_MAP = {
    'STATIC_PED':         'Pedestrian\n(Hard)',
    'DYNAMIC_PED':        'Pedestrian\n(Hard)',
    'STATIC_DOOR_OPEN':   'Door\n(Hard)',
    'DYNAMIC_DOOR_OPEN':  'Door\n(Hard)',
    'STATIC_COLLIDE':     'Collision\n(Medium)',
    'DYNAMIC_COLLIDE':    'Collision\n(Medium)',
    'STATIC_MISS':        'Miss\n(Medium)',
    'DYNAMIC_MISS':       'Miss\n(Medium)',
    'STATIC_STOP_EARLY':  'Stop\n(Medium)',
    'DYNAMIC_STOP_EARLY': 'Stop\n(Medium)',
}


def load_final_results(results_dir):
    pattern = os.path.join(results_dir, '*', 'results.json')
    paths = sorted(glob.glob(pattern))

    static_labels, static_ious, static_cols = [], [], []
    dynamic_labels, dynamic_ious, dynamic_cols = [], [], []

    for path in paths:
        folder = os.path.basename(os.path.dirname(path))
        suffix = re.sub(r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_(?:FINAL_FINAL_)?', '', folder)
        label = _LABEL_MAP.get(suffix, suffix)

        with open(path) as f:
            data = json.load(f)

        iou = data['mean_iou']
        col = data['actual_collision_rate']

        if suffix.startswith('STATIC_'):
            static_labels.append(label)
            static_ious.append(iou)
            static_cols.append(col)
        elif suffix.startswith('DYNAMIC_'):
            dynamic_labels.append(label)
            dynamic_ious.append(iou)
            dynamic_cols.append(col)

    return (static_labels, static_ious, static_cols), \
           (dynamic_labels, dynamic_ious, dynamic_cols)


def plot_group(labels, ious, collision_rates, title, out_path, y_max=None):
    n = len(labels)
    if n == 0:
        print(f'No data for {title}, skipping.')
        return

    x = np.arange(n)
    bar_w = 0.35

    fig, ax = plt.subplots(figsize=(max(6, n * 2), 5))

    bars_iou = ax.bar(x - bar_w / 2, ious, bar_w,
                      label='Mean IOU', color='steelblue')
    bars_col = ax.bar(x + bar_w / 2, collision_rates, bar_w,
                      label='Collision Rate', color='tomato')

    for bar, val in zip(bars_iou, ious):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars_col, collision_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, y_max)
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f'Saved to {out_path}')
    plt.show()


if __name__ == '__main__':
    _here = os.path.dirname(os.path.abspath(__file__))
    _results_dir = os.path.join(_here, 'results', 'final_results')

    (sl, si, sc), (dl, di, dc) = load_final_results(_results_dir)

    # Compute shared y-axis max across both groups
    all_vals = si + sc + di + dc
    y_max = max(1.15, max(all_vals) * 1.15) if all_vals else 1.15

    plot_group(sl, si, sc, 'Static Scenarios — IOU and Collision Rate', OUT_STATIC, y_max=y_max)
    plot_group(dl, di, dc, 'Dynamic Scenarios — IOU and Collision Rate', OUT_DYNAMIC, y_max=y_max)