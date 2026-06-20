#!/usr/bin/env python3
"""create_parking_buckets_v3.py

Builds parking_buckets_v3 from the existing v1 pkl by splitting EVERY major
maneuver bucket by episode type (freeroll / ped / door) so the datamodule can
weight each independently.  v2 only split pre_turn, turn_execution and
swing_out; v3 extends the same split to approach, final_turn, into_spot,
recovery_park and correction_maneuver, because those buckets are still
majority-pedestrian (e.g. approach 52% ped, recovery_park 59% ped) and were
letting pedestrian frames dominate training despite the v2 ped downweighting.

For each base bucket B in SPLIT_BUCKETS we add:
    B_freeroll  — frames from cone + empty episodes (approach at speed)
    B_ped       — frames from pedestrian episodes (stop-then-go dynamic)
    B_door      — frames from door episodes

turn_execution is split too, but per the v2 rationale only turn_exec_freeroll
is meant to be sampled (stopped-ped turn_execution frames taught late turns).

All original v1 buckets are preserved unchanged so any yaml can still
reference them ('all' in particular is kept as the background bucket).

Usage:
    python create_parking_buckets_v3.py [--v1-dir DIR] [--out-dir DIR]
"""

import argparse
import collections
import gzip
import json
import pathlib
import pickle

REPO_ROOT = pathlib.Path(
    "/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo"
).resolve()

DEFAULT_V1 = pathlib.Path(
    "/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter"
    "/finetune/parking_buckets_v1"
)
DEFAULT_OUT = pathlib.Path(
    "/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter"
    "/finetune/parking_buckets_v3"
)

CONE_TYPES  = {"normal", "normal_close", "recovery"}
EMPTY_TYPES = {"normal_empty", "normal_close_empty"}
PED_TYPES   = {"pedestrian_normal", "pedestrian_recovery"}
DOOR_TYPES  = {"door_normal"}
FREEROLL    = CONE_TYPES | EMPTY_TYPES   # approaches at speed, no forced stop

# Every maneuver bucket we want to weight per episode type.
SPLIT_BUCKETS = (
    "approach",
    "swing_out",
    "final_turn",
    "into_spot",
    "pre_turn",
    "turn_execution",
    "recovery_park",
    "correction_maneuver",
)


def _meta(rel_path: str) -> dict:
    p = REPO_ROOT / rel_path
    meta = p.parent.parent / "episode_meta.json"
    if meta.exists():
        return json.load(open(meta))
    return {}


def group(ep_type: str) -> str:
    if ep_type in FREEROLL:   return "freeroll"
    if ep_type in PED_TYPES:  return "ped"
    if ep_type in DOOR_TYPES: return "door"
    return "unk"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v1-dir",  type=pathlib.Path, default=DEFAULT_V1)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    v1_pkl = args.v1_dir / "buckets_paths.pkl"
    print(f"Loading v1 pkl from {v1_pkl}")
    v1: dict[str, list[str]] = pickle.load(open(v1_pkl, "rb"))

    # Start with a full copy of v1 (keeps 'all' + the unsplit originals).
    v3: dict[str, list[str]] = {k: list(v) for k, v in v1.items()}

    # episode_type cache keyed by rel path.
    ep_cache: dict[str, str] = {}
    def cached_group(rel: str) -> str:
        if rel not in ep_cache:
            ep_cache[rel] = group(_meta(rel).get("episode_type", "unk"))
        return ep_cache[rel]

    print("Splitting every major bucket by episode type (freeroll/ped/door) …")
    for base in SPLIT_BUCKETS:
        for suffix in ("freeroll", "ped", "door"):
            v3[f"{base}_{suffix}"] = []
        for rel in v1[base]:
            g = cached_group(rel)
            if g in ("freeroll", "ped", "door"):
                v3[f"{base}_{g}"].append(rel)

    # Split the background 'all' bucket by episode type too.  'all' is ~48% ped,
    # so if it is sampled as one mixed bucket it drags the global freeroll:ped
    # mix off the per-maneuver 2:1.  Splitting lets the yaml hold the background
    # at 2:1 as well.  The original 'all' bucket is kept (validation runs on it
    # and the sampler references it) but is weighted minimally in the yaml.
    for suffix in ("freeroll", "ped", "door"):
        v3[f"all_{suffix}"] = []
    for rel in v1["all"]:
        g = cached_group(rel)
        if g in ("freeroll", "ped", "door"):
            v3[f"all_{g}"].append(rel)

    # Speed-split turn_execution_ped: keep only the MOVING frames (car actually
    # driving through the turn) as turn_execution_ped_moving.  The stopped /
    # waiting-for-pedestrian frames (speed <= SPEED_THRESH) at 0-8 m are the ones
    # that looked like straight-approach and taught late turn initiation, so they
    # are deliberately left out of the moving bucket.
    SPEED_THRESH = 0.5  # m/s
    def _speed(rel: str) -> float:
        try:
            return float(json.load(gzip.open(REPO_ROOT / rel)).get("speed", 0.0))
        except Exception:
            return 0.0
    v3["turn_execution_ped_moving"] = [
        r for r in v3["turn_execution_ped"] if _speed(r) > SPEED_THRESH
    ]
    print(f"turn_execution_ped_moving (speed > {SPEED_THRESH} m/s): "
          f"{len(v3['turn_execution_ped_moving'])} of {len(v3['turn_execution_ped'])} ped frames")

    # ------------------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_pkl = args.out_dir / "buckets_paths.pkl"
    pickle.dump(v3, open(out_pkl, "wb"))
    print(f"\nSaved v3 pkl → {out_pkl}")

    # ------------------------------------------------------------------
    total = len(v3["all"])
    print(f"\n{'='*72}")
    print(f"BUCKET COMPOSITION  (total frames in 'all': {total})")
    print(f"{'='*72}")
    print(f"{'bucket':<28} {'total':>6}  {'freeroll':>10}  {'ped':>9}  {'door':>7}")
    print(f"{'-'*72}")
    stats = {}
    for name in sorted(v3):
        paths = v3[name]
        gc = collections.Counter(cached_group(r) for r in paths)
        n = len(paths)
        pct = lambda x: f"{100*x/max(n,1):.0f}%"
        print(f"{name:<28} {n:>6}  "
              f"{gc['freeroll']:>5}({pct(gc['freeroll'])})  "
              f"{gc['ped']:>4}({pct(gc['ped'])})  "
              f"{gc['door']:>3}({pct(gc['door'])})")
        stats[name] = {"total": n, "freeroll": gc['freeroll'],
                       "ped": gc['ped'], "door": gc['door'], "unk": gc['unk']}

    json.dump(stats, open(args.out_dir / "buckets_stats.json", "w"), indent=2)
    print(f"\nSaved stats → {args.out_dir / 'buckets_stats.json'}")


if __name__ == "__main__":
    main()
