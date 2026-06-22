"""
AutoVLA-style "disk diagram" of training-trajectory / goal-position coverage.

Reproduces the spirit of AutoVLA supplementary Fig. S1: a bird's-eye-view of
trajectory endpoints drawn as short oriented dashes (position = where the
trajectory ends in ego frame, orientation = its heading), so you can see which
maneuvers the dataset actually teaches. We overlay the goal positions
(`target_point`) to answer the reviewer's question: does the training set cover
the spectrum of goal positions we ask the model to reach?

Ego frame convention (carla_garage / SimLingo): x = forward, y = lateral,
ego at the origin. route[0] == (0,0).

Run (simlingo env):
    python plot_goal_coverage.py \
        --data run_001/data/simlingo/parking_ft/routes_training \
        --out goal_coverage_disk.png
"""

import argparse
import glob
import gzip
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_episode(meas_dir, frame_stride):
    """Yield (endpoint_xy, heading_rad, is_reverse, target_point_xy) per sampled frame."""
    files = sorted(glob.glob(os.path.join(meas_dir, "*.json.gz")))
    for f in files[::frame_stride]:
        try:
            d = json.load(gzip.open(f))
        except (OSError, EOFError, json.JSONDecodeError):
            continue  # skip a corrupt/partial file rather than abort the whole scan
        route = np.asarray(d.get("route", []), dtype=float)
        if route.shape[0] < 2:
            continue
        end = route[-1]
        # heading from the last meaningful segment (use a few points back for stability)
        prev = route[-3] if route.shape[0] >= 3 else route[-2]
        seg = end - prev
        heading = np.arctan2(seg[1], seg[0])
        # reverse if the trajectory's net forward motion is backward
        is_reverse = (route[-1, 0] - route[0, 0]) < -0.5
        tp = np.asarray(d.get("target_point", [np.nan, np.nan]), dtype=float)
        yield route, end, heading, is_reverse, tp


def clip_to_horizon(route, horizon_m):
    """Clip a polyline to the first `horizon_m` metres of arc length from the start.

    This is the key step to make the plot look like AutoVLA Fig. S1: instead of
    drawing the full variable-length route-to-goal (0..~50 m), every trajectory is
    truncated to a common look-ahead, so they form a coherent fan.
    """
    seglen = np.hypot(*np.diff(route, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    if cum[-1] <= horizon_m:
        return route
    last = int(np.searchsorted(cum, horizon_m))           # first vertex past horizon
    pts = route[:last]
    # interpolate the exact crossing point so every clipped route ends at horizon_m
    over = (horizon_m - cum[last - 1]) / seglen[last - 1]
    cross = route[last - 1] + over * (route[last] - route[last - 1])
    return np.vstack([pts, cross])


def collect_primitives(data_root, frames_per_segment):
    """Per-step motion primitives (dx, dy, dtheta) over a fixed time window.

    This is the continuous analog of AutoVLA's discrete action codebook (their
    Fig. S1): each primitive is the relative SE(2) pose between frame i and
    frame i+frames_per_segment, expressed in frame i's coordinate -
    i.e. (dx, dy, dtheta) = "what the car does in the next 0.5 s". We skip the
    K-Disk clustering / 2048-token deduplication because this model regresses
    waypoints continuously and never quantizes actions; the raw density is the
    honest equivalent.
    """
    dxs, dys, dthetas = [], [], []
    epi_dirs = sorted(glob.glob(os.path.join(data_root, "**", "measurements"), recursive=True))
    for meas_dir in epi_dirs:
        files = sorted(glob.glob(os.path.join(meas_dir, "*.json.gz")))
        mats = []
        for f in files:
            try:
                mats.append(np.asarray(json.load(gzip.open(f))["ego_matrix"], float))
            except (OSError, EOFError, json.JSONDecodeError, KeyError):
                mats.append(None)
        for i in range(len(mats) - frames_per_segment):
            mi, mj = mats[i], mats[i + frames_per_segment]
            if mi is None or mj is None:
                continue
            t = np.linalg.inv(mi) @ mj            # pose of frame j in frame i's coords
            dxs.append(t[0, 3])
            dys.append(t[1, 3])
            dthetas.append(np.arctan2(t[1, 0], t[0, 0]))
    print(f"Collected {len(dxs)} motion primitives "
          f"({frames_per_segment} frames = {frames_per_segment*0.25:.2f}s window)")
    return np.array(dxs), np.array(dys), np.array(dthetas)


def collect(data_root, frame_stride):
    routes, ends, headings, reverses, targets = [], [], [], [], []
    epi_dirs = sorted(glob.glob(os.path.join(data_root, "**", "measurements"), recursive=True))
    print(f"Found {len(epi_dirs)} episodes under {data_root}")
    for meas_dir in epi_dirs:
        for route, end, heading, is_rev, tp in load_episode(meas_dir, frame_stride):
            routes.append(route)
            ends.append(end)
            headings.append(heading)
            reverses.append(is_rev)
            targets.append(tp)
    ends = np.array(ends)
    headings = np.array(headings)
    reverses = np.array(reverses, dtype=bool)
    targets = np.array(targets)
    print(f"Collected {len(ends)} sampled frames "
          f"({reverses.sum()} reverse, {(~reverses).sum()} forward)")
    return routes, ends, headings, reverses, targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="run_001/data/simlingo/parking_ft/routes_training")
    ap.add_argument("--out", default="goal_coverage_disk.png")
    ap.add_argument("--frame-stride", type=int, default=5,
                    help="sample every Nth frame per episode to keep the plot readable")
    ap.add_argument("--horizon", type=float, default=12.0,
                    help="look-ahead in metres each trajectory is clipped to for the fan (panel a)")
    ap.add_argument("--segment-frames", type=int, default=2,
                    help="frames per motion primitive; 2 frames x 0.25s = 0.5s (AutoVLA window)")
    args = ap.parse_args()

    routes, ends, _headings, reverses, _targets = collect(args.data, args.frame_stride)
    if len(ends) == 0:
        raise SystemExit("No trajectories found — check --data path.")
    pdx, pdy, pdth = collect_primitives(args.data, args.segment_frames)

    fig = plt.figure(figsize=(16, 7))
    ax_a = fig.add_subplot(1, 2, 1)
    ax_d = fig.add_subplot(1, 2, 2)

    # ---- (a) AutoVLA-style fan: trajectory polylines clipped to a common horizon ----
    for route, is_rev in zip(routes, reverses):
        clipped = clip_to_horizon(route, args.horizon)
        ax_a.plot(clipped[:, 0], clipped[:, 1],
                  color="crimson" if is_rev else "0.45",
                  alpha=0.06, lw=0.8, solid_capstyle="round")
    ax_a.scatter([0], [0], c="black", s=60, marker="^", zorder=10, label="ego")
    ax_a.set_title(f"(a) Trajectory fan, clipped to {args.horizon:.0f} m  |  n={len(routes)}\n"
                   "grey=forward, red=reverse")
    ax_a.set_xlabel("forward x [m]")
    ax_a.set_ylabel("lateral y [m]")
    ax_a.set_aspect("equal")
    lim = args.horizon * 1.1
    ax_a.set_xlim(-lim, lim)
    ax_a.set_ylim(-lim, lim)
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(loc="upper right")

    # ---- (b) raw motion-primitive arrows: every 0.5 s (dx, dy, dtheta) drawn ----
    # Same arrow style as the original 'codebook', but NO deduplication and NO
    # frequency colour - every individual primitive gets its own arrow, positioned
    # at (dx, dy) and pointing along dtheta. Dense regions (slow creep) darken
    # naturally via overplotting at low alpha.
    ax_d.quiver(pdx, pdy, np.cos(pdth), np.sin(pdth),
                angles="xy", scale=30, width=0.0018, headwidth=4,
                color="0.2", alpha=0.05)
    ax_d.scatter([0], [0], c="red", s=60, marker="^", zorder=10)
    ax_d.set_title(f"(b) Raw motion primitives  |  {len(pdx)} arrows\n"
                   "0.5 s (dx, dy, d-theta)")
    ax_d.set_xlabel("dx forward [m]")
    ax_d.set_ylabel("dy lateral [m]")
    ax_d.set_aspect("equal")
    ax_d.grid(True, alpha=0.3)

    fig.suptitle("Training coverage disk - does the data span the goal spectrum?",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
