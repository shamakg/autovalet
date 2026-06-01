"""Revert the pedestrian-waiting route-zeroing in collected data (in-place).

Background
----------
collect_data.py used to collapse the route to all-zeros whenever the ego was
yielding to a dynamic obstacle (``scenario.car.car.waiting``). That corrupted
the *route/path* supervision: the path should always point toward the
destination. The *speed* waypoints (data['waypoints']) are derived separately
in the dataloader from future ego_matrix poses, so they already cluster at the
origin when the ego is stopped and need no change here.

This script reconstructs the route for waiting frames from the nearest
non-waiting frame, transformed into the waiting frame's ego coordinates via the
per-frame ego_matrix (the same ego->world->ego convention the dataloader's
get_waypoints uses). It rewrites both ``route`` and ``route_original``.

Detection: a waiting frame is one where ALL 20 route points are [0, 0].
Normal frames only have route[0] == [0, 0] with the rest non-zero.

Usage:
    python postprocess_waiting_route.py --dry-run      # report only
    python postprocess_waiting_route.py --apply         # rewrite in place
"""

import argparse
import glob
import gzip
import json
import os

import numpy as np

DATA_DIR = (
    "/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/"
    "finetune/run_001/data/simlingo/parking_ft/routes_training/"
    "RouteScenario_parking"
)
NUM_POINTS = 20


def is_waiting(route):
    return np.allclose(np.asarray(route, dtype=float), 0.0)


def mat_rt(ego_matrix):
    """Return (R 3x3, t 3,) from a 4x4 ego->world matrix."""
    m = np.asarray(ego_matrix, dtype=float)[:3]
    return m[:, :3], m[:, 3]


def ego_to_world(route_xy, R, t):
    """route_xy: (N,2) in ego frame (x fwd, y left, z=0) -> (N,3) world."""
    pts = np.asarray(route_xy, dtype=float)
    pts3 = np.concatenate([pts, np.zeros((len(pts), 1))], axis=1)  # (N,3)
    return (R @ pts3.T).T + t  # (N,3)


def world_to_ego(world_xyz, R, t):
    """world (N,3) -> (N,2) in ego frame."""
    rel = np.asarray(world_xyz, dtype=float) - t
    ego = (R.T @ rel.T).T  # (N,3)
    return ego[:, :2]


def reanchor(route_ego, target_pt):
    """Mirror collect_data: anchor route[0] at ego origin, sample 19 forward.

    Handles the decel-into-stop frames where the ego crept along the path and
    some transformed points now sit behind the origin.
    """
    pts = np.asarray(route_ego, dtype=float)
    dists = np.linalg.norm(pts, axis=1)
    ti = int(np.argmin(dists))
    forward = pts[ti:]
    if len(forward) > 1:
        rest = forward[1:]
        idx = np.linspace(0, len(rest) - 1, NUM_POINTS - 1, dtype=int)
        out = np.vstack([[0.0, 0.0], rest[idx]])
    else:
        out = np.linspace([0.0, 0.0], target_pt, NUM_POINTS)
    return out.tolist()


def process_episode(ep_dir, apply):
    files = sorted(glob.glob(os.path.join(ep_dir, "measurements", "*.json.gz")))
    if not files:
        return 0, 0

    # Load all frames once.
    metas = []
    for f in files:
        with gzip.open(f, "rt") as fh:
            metas.append(json.load(fh))

    waiting_idx = [i for i, d in enumerate(metas) if is_waiting(d["route"])]
    if not waiting_idx:
        return 0, 0

    nonwait_idx = [i for i, d in enumerate(metas) if not is_waiting(d["route"])]
    fixed = 0

    for i in waiting_idx:
        # Nearest non-waiting frame: prefer preceding, else following.
        prev = [j for j in nonwait_idx if j < i]
        nxt = [j for j in nonwait_idx if j > i]
        if prev:
            src = prev[-1]
        elif nxt:
            src = nxt[0]
        else:
            src = None  # whole episode waiting -> straight-line fallback

        d = metas[i]
        Rw, tw = mat_rt(d["ego_matrix"])

        if src is not None:
            s = metas[src]
            Rs, ts = mat_rt(s["ego_matrix"])
            world = ego_to_world(s["route"], Rs, ts)
            route_ego = world_to_ego(world, Rw, tw)
            new_route = reanchor(route_ego, d["target_point"])
        else:
            new_route = np.linspace(
                [0.0, 0.0], d["target_point"], NUM_POINTS
            ).tolist()

        d["route"] = new_route
        d["route_original"] = new_route
        fixed += 1

    if apply:
        # Only rewrite the waiting frames we actually changed.
        for i in waiting_idx:
            with gzip.open(files[i], "wt", encoding="utf-8") as fh:
                json.dump(metas[i], fh)

    return fixed, len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    if not (args.dry_run or args.apply):
        ap.error("pass --dry-run or --apply")

    eps = sorted(glob.glob(os.path.join(args.data_dir, "Town04_*")))
    total_fixed = 0
    eps_touched = 0
    samples = []
    for ep in eps:
        fixed, _ = process_episode(ep, apply=args.apply)
        if fixed:
            eps_touched += 1
            total_fixed += fixed
            if len(samples) < 3:
                samples.append((os.path.basename(ep), fixed))

    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"[{mode}] episodes={len(eps)} touched={eps_touched} "
          f"waiting_frames_fixed={total_fixed}")
    print("samples (episode, frames_fixed):", samples)


if __name__ == "__main__":
    main()
