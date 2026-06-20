#!/usr/bin/env python3
"""create_parking_buckets_v2.py

Builds parking_buckets_v2 from the existing v1 pkl by splitting the two
turn-critical buckets (pre_turn, turn_execution) by episode type so the
datamodule can weight cone/empty and pedestrian frames independently.

New buckets added on top of v1:
  pre_turn_freeroll       — pre_turn frames from cone + empty episodes
  pre_turn_ped            — pre_turn frames from pedestrian episodes
  pre_turn_door           — pre_turn frames from door episodes
  turn_exec_freeroll      — turn_execution frames from cone + empty episodes
                            (ped turn_execution was deliberately excluded in v1
                             because stopped-ped frames taught the model to delay)

All existing v1 buckets are preserved unchanged so the yaml can still
reference them.

Usage:
    python create_parking_buckets_v2.py [--v1-dir DIR] [--out-dir DIR]
"""

import argparse
import collections
import json
import pathlib
import pickle

REPO_ROOT = pathlib.Path(
    "/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo"
).resolve()

DEFAULT_V1  = pathlib.Path(
    "/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter"
    "/finetune/parking_buckets_v1"
)
DEFAULT_OUT = pathlib.Path(
    "/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter"
    "/finetune/parking_buckets_v2"
)

CONE_TYPES  = {"normal", "normal_close", "recovery"}
EMPTY_TYPES = {"normal_empty", "normal_close_empty"}
PED_TYPES   = {"pedestrian_normal", "pedestrian_recovery"}
DOOR_TYPES  = {"door_normal"}
FREEROLL    = CONE_TYPES | EMPTY_TYPES   # approaches at speed, no forced stop


def episode_type(rel_path: str) -> str:
    p = REPO_ROOT / rel_path
    meta = p.parent.parent / "episode_meta.json"
    if meta.exists():
        return json.load(open(meta)).get("episode_type", "unk")
    return "unk"


def group(ep_type: str) -> str:
    if ep_type in FREEROLL: return "freeroll"
    if ep_type in PED_TYPES:  return "ped"
    if ep_type in DOOR_TYPES: return "door"
    return "unk"


def destination_of(rel_path: str) -> int:
    p = REPO_ROOT / rel_path
    meta = p.parent.parent / "episode_meta.json"
    if meta.exists():
        return json.load(open(meta)).get("destination", -1)
    return -1


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v1-dir",  type=pathlib.Path, default=DEFAULT_V1)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    v1_pkl = args.v1_dir / "buckets_paths.pkl"
    print(f"Loading v1 pkl from {v1_pkl}")
    v1: dict[str, list[str]] = pickle.load(open(v1_pkl, "rb"))

    # Start with a full copy of v1
    v2: dict[str, list[str]] = {k: list(v) for k, v in v1.items()}

    # Add the new split buckets
    for key in ("pre_turn_freeroll", "pre_turn_ped", "pre_turn_door",
                "turn_exec_freeroll",
                "swing_out_freeroll", "swing_out_ped", "swing_out_door"):
        v2[key] = []

    print("Splitting pre_turn, turn_execution, and swing_out by episode type …")
    ep_cache: dict[str, str] = {}

    def cached_ep_type(rel: str) -> str:
        if rel not in ep_cache:
            ep_cache[rel] = episode_type(rel)
        return ep_cache[rel]

    for rel in v1["pre_turn"]:
        g = group(cached_ep_type(rel))
        if   g == "freeroll": v2["pre_turn_freeroll"].append(rel)
        elif g == "ped":      v2["pre_turn_ped"].append(rel)
        elif g == "door":     v2["pre_turn_door"].append(rel)

    for rel in v1["turn_execution"]:
        if group(cached_ep_type(rel)) == "freeroll":
            v2["turn_exec_freeroll"].append(rel)

    # swing_out: 57% ped in v1 — same skew as pre_turn, same fix.
    # Freeroll swing_out has different entry velocity and arc geometry vs
    # post-ped-stop swing_out, so they need independent sampling weights.
    for rel in v1["swing_out"]:
        g = group(cached_ep_type(rel))
        if   g == "freeroll": v2["swing_out_freeroll"].append(rel)
        elif g == "ped":      v2["swing_out_ped"].append(rel)
        elif g == "door":     v2["swing_out_door"].append(rel)

    # ------------------------------------------------------------------
    # Save pkl
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_pkl = args.out_dir / "buckets_paths.pkl"
    pickle.dump(v2, open(out_pkl, "wb"))
    print(f"\nSaved v2 pkl → {out_pkl}")

    # ------------------------------------------------------------------
    # Full composition report
    total = len(v2["all"])
    print(f"\n{'='*72}")
    print(f"BUCKET COMPOSITION  (total frames in 'all': {total})")
    print(f"{'='*72}")
    print(f"{'bucket':<24} {'total':>6}  {'freeroll':>10}  {'ped':>8}  {'door':>6}  {'unk':>5}")
    print(f"{'-'*72}")

    stats = {}
    for name, paths in sorted(v2.items()):
        grp_counts: dict[str, int] = collections.Counter(
            group(cached_ep_type(r)) for r in paths
        )
        n = len(paths)
        fr = grp_counts["freeroll"]
        pd = grp_counts["ped"]
        dr = grp_counts["door"]
        uk = grp_counts["unk"]
        pct = lambda x: f"{100*x/max(n,1):.0f}%"
        print(f"{name:<24} {n:>6}  "
              f"{fr:>5}({pct(fr)})  "
              f"{pd:>4}({pct(pd)})  "
              f"{dr:>3}({pct(dr)})  "
              f"{uk:>3}")
        stats[name] = {"total": n, "freeroll": fr, "ped": pd, "door": dr, "unk": uk}

    # ------------------------------------------------------------------
    # Per-destination scenario split for the critical new buckets
    print(f"\n{'='*72}")
    print("PER-DESTINATION BREAKDOWN  (new buckets only)")
    print(f"{'='*72}")

    dest_cache: dict[str, int] = {}
    def cached_dest(rel: str) -> int:
        if rel not in dest_cache:
            dest_cache[rel] = destination_of(rel)
        return dest_cache[rel]

    for bucket in ("pre_turn_freeroll", "pre_turn_ped", "pre_turn_door",
                   "turn_exec_freeroll",
                   "swing_out_freeroll", "swing_out_ped", "swing_out_door"):
        paths = v2[bucket]
        dest_counts: dict[int, int] = collections.Counter(
            cached_dest(r) for r in paths
        )
        print(f"\n  {bucket}  (total={len(paths)})")
        for dest in sorted(dest_counts):
            print(f"    dest={dest:>3}: {dest_counts[dest]:>4} frames")

    # ------------------------------------------------------------------
    # Save stats json
    stats_path = args.out_dir / "buckets_stats.json"
    json.dump(stats, open(stats_path, "w"), indent=2)
    print(f"\nSaved stats → {stats_path}")

    # ------------------------------------------------------------------
    # Print suggested yaml weights
    pt_fr = len(v2["pre_turn_freeroll"])
    pt_pd = len(v2["pre_turn_ped"])
    pt_dr = len(v2["pre_turn_door"])
    te_fr = len(v2["turn_exec_freeroll"])
    so_fr = len(v2["swing_out_freeroll"])
    so_pd = len(v2["swing_out_ped"])
    so_dr = len(v2["swing_out_door"])

    print(f"\n{'='*72}")
    print("SUGGESTED simlingo_seed1.yaml  train_partitions  (equalises freeroll/ped)")
    print(f"{'='*72}")
    print("""  train_partitions:
    all:                  0.05
    approach:             0.10
    # swing_out split — entry velocity/arc geometry differs at freeroll speed vs post-ped-stop
    swing_out_freeroll:   0.35   # {so_fr} frames (cone+empty)  ← upsampled to match ped
    swing_out_ped:        0.35   # {so_pd} frames (pedestrian)
    swing_out_door:       0.05   # {so_dr} frames (door) — keep minor
    final_turn:           0.35
    into_spot:            0.40
    # pre_turn split by episode type — equal group weight regardless of raw count
    pre_turn_freeroll:    0.35   # {pt_fr} frames (cone+empty)  ← upsampled to match ped
    pre_turn_ped:         0.35   # {pt_pd} frames (pedestrian)
    pre_turn_door:        0.05   # {pt_dr} frames (door) — keep minor
    # turn_exec_freeroll: cone/empty only — adds back the approaching-at-speed
    #   0-8 m signal without the stopped-ped frames that caused turn-delay
    turn_exec_freeroll:   0.20   # {te_fr} frames
    recovery_park:        0.15
    correction_maneuver:  0.30""".format(
        pt_fr=pt_fr, pt_pd=pt_pd, pt_dr=pt_dr, te_fr=te_fr,
        so_fr=so_fr, so_pd=so_pd, so_dr=so_dr))


if __name__ == "__main__":
    main()
