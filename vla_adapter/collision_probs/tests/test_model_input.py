#!/usr/bin/env python3
"""test_model_input.py

Run a SINGLE CARLA parking episode and dump the EXACT topdown collision-heatmap
view the VLA model receives each frame (the second model tile), so you can
eyeball what the heatmap input actually looks like before a full re-train.

For every frame it saves, to --out:
  XXXX_topdown.jpg    the topdown collision-heatmap overlay supplied to the model
                      as the second view (topdown camera + green->red risk).
  XXXX_tile0.jpg      the InternVL2 tile the vision encoder actually sees for the
                      topdown view, after tick() preprocessing (dynamic_preprocess,
                      max_num=1, no cut_bottom_quarter).
At the end it writes:
  model_input.mp4     per-frame [topdown view | encoder tile] composite video
  contact_sheet.jpg   the same layout as a still, for a few frames.

Requires CARLA running on :2000 and the same env as benchmark.py (see
bash_scripts/run_benchmark.sh). Example:
    python collision_probs/test_model_input.py
    python collision_probs/test_model_input.py --mode collide

--no-model: drive with the A* expert and do NOT load the VLA model (fast). The
corridor heatmap is then clipped by the A* route -- exactly the training-time
input. Reuses collect_data_topdown.collect_episode and reads the topdown_heatmap/
frames it writes (the model's second view). Episode type is the collect-style
name, e.g.:
    python collision_probs/test_model_input.py --no-model --episode-type pedestrian_normal
    python collision_probs/test_model_input.py --no-model --episode-type opposite_collide
"""
import argparse
import glob
import os
import sys
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

_COLLISION_PROBS = os.path.dirname(os.path.abspath(__file__))
_VLA_ADAPTER = os.path.dirname(_COLLISION_PROBS)
for _p in (_VLA_ADAPTER, _COLLISION_PROBS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from benchmark import run_scenario, _DEFAULT_CHECKPOINT
from default_runner import ScenarioMode
from v2_experiment import SCENARIOS
from testbed.v2_experiment_utils import load_client, town04_load, town04_spectator_bev
from heatmap_agent import HeatmapSimLingoAdapter


def _compose_row(topdown_path, row_h=384, pad=6):
    """One [topdown view | tile0 | ...] strip (PIL RGB), all scaled to row_h."""
    stem = topdown_path[:-len('_topdown.jpg')]
    s_img = Image.open(topdown_path).convert('RGB')
    tiles = [Image.open(t).convert('RGB')
             for t in sorted(glob.glob(stem + '_tile*.jpg'))]
    parts = [s_img] + tiles
    scaled = [p.resize((max(1, int(p.width * row_h / p.height)), row_h)) for p in parts]
    row_w = sum(p.width + pad for p in scaled) - pad
    row = Image.new('RGB', (row_w, row_h), (20, 20, 28))
    x = 0
    for p in scaled:
        row.paste(p, (x, 0)); x += p.width + pad
    return row


def _make_video(out_dir, fps=10, row_h=384):
    """Per-frame [stitched | encoder tiles] composite -> model_input.mp4."""
    stitched = sorted(glob.glob(os.path.join(out_dir, '*_stitched.jpg')))
    if not stitched:
        print("[video] no frames captured")
        return
    first = _compose_row(stitched[0], row_h)
    W, H = first.size
    out = os.path.join(out_dir, 'model_input.mp4')
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    for k, sp in enumerate(stitched):
        row = _compose_row(sp, row_h)
        if row.size != (W, H):                       # guard against odd frame
            row = row.resize((W, H))
        frame = cv2.cvtColor(np.asarray(row), cv2.COLOR_RGB2BGR)
        cv2.putText(frame, f'frame {k}', (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    print(f"[video] {out}  ({len(stitched)} frames, left=stitched rgb/, right=encoder tiles)")


def _contact_sheet(out_dir, max_frames=4, row_h=384):
    """Stack a few frames' [stitched | tiles] strips into one still."""
    stitched = sorted(glob.glob(os.path.join(out_dir, '*_stitched.jpg')))
    if not stitched:
        return
    idxs = np.linspace(0, len(stitched) - 1, min(max_frames, len(stitched))).astype(int)
    rows = [_compose_row(stitched[k], row_h) for k in idxs]
    pad = 6
    W = max(r.width for r in rows)
    H = sum(r.height for r in rows) + pad * (len(rows) - 1)
    sheet = Image.new('RGB', (W, H), (20, 20, 28))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y)); y += r.height + pad
    out = os.path.join(out_dir, 'contact_sheet.jpg')
    sheet.save(out, quality=95)
    print(f"[contact sheet] {out}")


def _setup_world():
    client = load_client()
    world = town04_load(client)
    CarlaDataProvider.set_client(client)
    CarlaDataProvider.set_world(world)
    town04_spectator_bev(world)
    return world


def run_with_model(args, out_dir):
    """Drive with the real VLA model (corridor clipped by the model's prediction)."""
    world = _setup_world()
    destination, parked_spots = SCENARIOS[args.scenario_index]

    def adapter_init_fn(a):
        a.heatmap_mode = args.heatmap_mode
        a.start_model_input_capture(out_dir)

    run_scenario(
        world, destination, parked_spots,
        ious=[], weighted_ious=[], collisions_ref=[0], near_misses_ref=[0],
        actual_collisions=[], walker_collisions_ref=[0],
        recording_path=None, car_list=[None],
        scenario_mode=ScenarioMode(args.mode),
        checkpoint_path=args.checkpoint,
        adapter_class=HeatmapSimLingoAdapter,
        adapter_init_fn=adapter_init_fn,
    )


def run_no_model(args, out_dir):
    """Drive with the A* expert, no VLA model loaded. Reuses collect_episode so the
    baked rgb/ frames ARE the model input (corridor clipped by the A* route)."""
    import collect_data_topdown as cdt
    from heatmap_agent import save_model_input_frame

    cdt.BAKE_MODE = args.heatmap_mode   # module global read by _compute_heatmaps

    world = _setup_world()
    settings = world.get_settings()
    settings.fixed_delta_seconds = cdt._FIXED_DT
    world.apply_settings(settings)

    destination, parked_spots = SCENARIOS[args.scenario_index]
    ep_dir = os.path.join(out_dir, 'episode')
    ok = cdt.collect_episode(world, ep_dir, args.episode_type, destination, parked_spots)
    if not ok or not os.path.isdir(os.path.join(ep_dir, 'rgb')):
        print("[no-model] episode failed (expert collision / low IoU) or was discarded "
              "-- rerun, optionally with a different --scenario-index/--episode-type")
        return
    rgbs = sorted(glob.glob(os.path.join(ep_dir, 'rgb', '*.jpg')))
    for i, f in enumerate(rgbs):
        stitched = np.array(Image.open(f).convert('RGB'))   # baked = the model input
        save_model_input_frame(out_dir, i, stitched, use_global_img=False)
    print(f"[no-model] baked {len(rgbs)} frames")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', type=str, default='pedestrian',
                    choices=[m.value for m in ScenarioMode],
                    help="scenario type for the model-driven path (default: pedestrian)")
    ap.add_argument('--no-model', action='store_true',
                    help="drive with the A* expert, don't load the VLA model (fast); "
                         "corridor clipped by the A* route, matching training.")
    ap.add_argument('--episode-type', type=str, default='pedestrian_normal',
                    help="collect-style episode type for --no-model (e.g. pedestrian_normal, "
                         "opposite_collide, normal, door_normal).")
    ap.add_argument('--heatmap-mode', type=str, default='ego_topdown',
                    choices=['ego_topdown', 'topdown_only'])
    ap.add_argument('--checkpoint', type=str, default=_DEFAULT_CHECKPOINT)
    ap.add_argument('--scenario-index', type=int, default=0,
                    help="which SCENARIOS entry to run (default: 0)")
    ap.add_argument('--out', type=str, default=None,
                    help="output dir (default: results/model_input_test/<ts>)")
    ap.add_argument('--fps', type=int, default=10, help="model_input.mp4 frame rate")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(
        _VLA_ADAPTER, 'results', 'model_input_test',
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + f'_{args.heatmap_mode}'
        + ('_astar' if args.no_model else ''))
    os.makedirs(out_dir, exist_ok=True)
    tag = f"episode_type={args.episode_type}" if args.no_model else f"scenario={args.mode}"
    print(f"{'[A* expert, no model]' if args.no_model else '[VLA model]'}  "
          f"{tag}  heatmap_mode={args.heatmap_mode}  ->  {out_dir}")

    try:
        if args.no_model:
            run_no_model(args, out_dir)
        else:
            run_with_model(args, out_dir)
    finally:
        n = len(glob.glob(os.path.join(out_dir, '*_stitched.jpg')))
        print(f"Captured {n} frames")
        _make_video(out_dir, fps=args.fps)
        _contact_sheet(out_dir)


if __name__ == '__main__':
    main()
