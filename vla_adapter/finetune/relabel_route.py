"""Post-process existing run_001 frames: replace broken route labels with a
straight-line interpolation from ego (0,0) to target_point.

The original collect_data.py saved route by sampling the full A* trajectory
(traj[0..len-1]) including already-traversed (past) positions in world coords,
then converting to ego frame. At training time this created a "backward snake"
starting behind the car. This script rewrites route/route_original as a clean
straight line from current ego to destination.

Usage:
    python relabel_route.py [run_dir]
    # default run_dir = ./run_001
"""

import sys
import gzip
import ujson
import pathlib
import numpy as np

def relabel_dir(run_dir: pathlib.Path):
    meas_dirs = sorted(run_dir.glob("data/simlingo/parking_ft/routes_training/RouteScenario_parking/*/measurements"))
    if not meas_dirs:
        # Also check flat layout (in case different run structure)
        meas_dirs = sorted(run_dir.glob("**/measurements"))
    print(f"Found {len(meas_dirs)} episode measurement dirs under {run_dir}")

    total_frames = 0
    for meas_dir in meas_dirs:
        files = sorted(meas_dir.glob("*.json.gz"))
        for f in files:
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                data = ujson.load(fh)

            tp = data["target_point"]  # [x, y] destination in ego frame

            # Straight line from (0,0) to target_point, 20 evenly spaced points
            new_route = np.linspace([0.0, 0.0], tp, 20).tolist()

            data["route"] = new_route
            data["route_original"] = new_route

            with gzip.open(f, "wt", encoding="utf-8") as fh:
                ujson.dump(data, fh, indent=4)

            total_frames += 1

        if files:
            print(f"  {meas_dir.parent.name}: relabeled {len(files)} frames")

    print(f"\nDone. Relabeled {total_frames} frames total.")


def smoke_test(run_dir: pathlib.Path):
    """Print first and last frame route of one episode to verify."""
    meas_dirs = sorted(run_dir.glob("**/measurements"))
    if not meas_dirs:
        return
    meas_dir = meas_dirs[0]
    files = sorted(meas_dir.glob("*.json.gz"))
    if len(files) < 2:
        return
    for label, f in [("first", files[0]), ("last", files[-1])]:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            data = ujson.load(fh)
        route = data["route"]
        tp = data["target_point"]
        print(f"  [{label}] target_point={tp}  route[0]={route[0]}  route[-1]={route[-1]}")


if __name__ == "__main__":
    run_dir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).parent / "run_001"
    relabel_dir(run_dir)
    print("\nSmoke test (one episode):")
    smoke_test(run_dir)
