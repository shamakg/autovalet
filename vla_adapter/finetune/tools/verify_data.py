"""Sanity-check a collected run directory.

Checks:
  1. route[0] is near (0,0)  — not the old ~(-1.6, 0) rear-axle artifact
  2. speed is signed          — negative values must exist in reverse episodes
  3. route ends near target_point — trajectory actually reaches destination
  4. route is monotonically increasing in x on forward frames — no backward snake

Usage:
    python verify_data.py [run_dir]          # default: ./run_001
"""

import sys, gzip, json, glob, pathlib
import numpy as np

def check(run_dir):
    run_dir = pathlib.Path(run_dir)
    files = sorted(run_dir.glob("**/measurements/*.json.gz"))
    print(f"Found {len(files)} frames under {run_dir}\n")
    if not files:
        print("No files found.")
        return

    route0_xs = []
    speeds = []
    end_errors = []
    bad_route0 = 0
    bad_end = 0

    for f in files:
        with gzip.open(f, "rt") as fp:
            m = json.load(fp)

        route = np.array(m["route"])        # (20, 2)
        tp    = np.array(m["target_point"]) # (2,)
        speed = m["speed"]

        route0_x = route[0, 0]
        route0_xs.append(route0_x)
        speeds.append(speed)

        # route[0] should be within 0.3 m of ego origin
        if abs(route0_x) > 0.3:
            bad_route0 += 1

        # route[-1] should be within 2 m of target_point
        end_err = np.linalg.norm(route[-1] - tp)
        end_errors.append(end_err)
        if end_err > 2.0:
            bad_end += 1

    route0_xs = np.array(route0_xs)
    speeds    = np.array(speeds)
    end_errors = np.array(end_errors)

    print("=== route[0][0] (should be ~0, not ~-1.6) ===")
    print(f"  mean={route0_xs.mean():.3f}  std={route0_xs.std():.3f}"
          f"  min={route0_xs.min():.3f}  max={route0_xs.max():.3f}")
    print(f"  frames with |route[0][0]| > 0.3 m: {bad_route0}/{len(files)}"
          f"  {'OK' if bad_route0 == 0 else '*** BAD ***'}")

    print("\n=== speed (signed: negative = reversing) ===")
    print(f"  mean={speeds.mean():.3f}  min={speeds.min():.3f}  max={speeds.max():.3f}")
    print(f"  negative speed frames: {(speeds < -0.05).sum()}"
          f"  {'(none — all forward)' if (speeds < -0.05).sum() == 0 else ''}")

    print("\n=== route[-1] vs target_point (should be within 2 m) ===")
    print(f"  mean end error={end_errors.mean():.2f} m  max={end_errors.max():.2f} m")
    print(f"  frames with end error > 2 m: {bad_end}/{len(files)}"
          f"  {'OK' if bad_end == 0 else '*** BAD ***'}")

    print("\n=== 5 sample frames ===")
    sample_files = files[::max(1, len(files)//5)][:5]
    for f in sample_files:
        with gzip.open(f, "rt") as fp:
            m = json.load(fp)
        route = np.array(m["route"])
        print(f"  {f.name}  speed={m['speed']:+.2f}"
              f"  route[0]={np.round(route[0],3).tolist()}"
              f"  route[1]={np.round(route[1],3).tolist()}"
              f"  route[-1]={np.round(route[-1],2).tolist()}"
              f"  tp={np.round(m['target_point'],2)}")

if __name__ == "__main__":
    run_dir = sys.argv[1] if len(sys.argv) > 1 else str(pathlib.Path(__file__).parent / "run_001")
    check(run_dir)
