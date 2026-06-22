"""collect_data_topdown.py

Top-down (bird's-eye) + heatmap data collection for the LICOM pipeline.

Saves per episode:
  rgb/XXXX.jpg           — front camera (1024×512, plain, same as collect_data.py)
  rgb_augmented/XXXX.jpg — laterally-shifted augmented front camera
  topdown_heatmap/XXXX.jpg — BEV camera + colorized risk overlay (512×512, 3-ch),
                           the SECOND image tile fed to SimLingo (use_topdown=true)
  measurements/          — ego pose / route / command (with aug_translation/rotation)
  boxes/                 — parked-car + walker bounding boxes

  topdown/ and heatmap/ are written during collection as intermediate inputs
  to the topdown_heatmap overlay above, then deleted at the end of each
  episode by _compute_heatmaps() — the dataloader reads topdown_heatmap/
  exclusively, so they're not worth keeping on disk.

Quick test (CARLA must be running on :2000), via collect_data_topdown.sh:
    COLLECT_EPISODES="pedestrian_normal:3" bash collect_data_topdown.sh
"""
import gzip
import json
import os
import pathlib
import queue
import random
import shutil
import sys

import numpy as np
import carla
from PIL import Image
import ujson

_VLA_ADAPTER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FINETUNE = os.path.join(_VLA_ADAPTER, "finetune")
if _FINETUNE not in sys.path:
    sys.path.insert(0, _FINETUNE)

from collect_data import (
    SAVE_EVERY_N, get_offsets,
    save_measurement, save_boxes, save_gmm_label,
    setup_camera,
    _EPISODE_MODE_MAP, TRAIN_SCENARIOS, run_scenario,
)
from crossing_gmm import render_route_risk_grid, heatmap_on_topdown
from config import (HORIZON_SCALE as HEATMAP_HORIZON,
                    ROUTE_CORRIDOR_HALF_WIDTH,
                    TOPDOWN_HEIGHT, TOPDOWN_SIZE, TOPDOWN_FOV, RISK_GRID_SIZE,
                    STATIC_OBSTACLES_ENABLED, KEEP_TOPDOWN_BASE)

TOPDOWN_HALF = TOPDOWN_HEIGHT * np.tan(np.deg2rad(TOPDOWN_FOV / 2.0))
TOPDOWN_RES  = 1.0

_FIXED_DT = 0.05

_COLLISION_PROBS = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(_COLLISION_PROBS, "run_topdown_001"))

def setup_topdown_camera(world, vehicle, height=TOPDOWN_HEIGHT,
                         size=TOPDOWN_SIZE, fov=TOPDOWN_FOV):
    bp = world.get_blueprint_library().find('sensor.camera.rgb')
    bp.set_attribute('image_size_x', str(size))
    bp.set_attribute('image_size_y', str(size))
    bp.set_attribute('fov', str(fov))
    bp.set_attribute('sensor_tick', str(SAVE_EVERY_N * _FIXED_DT))
    transform = carla.Transform(
        carla.Location(x=0.0, y=0.0, z=height),
        carla.Rotation(pitch=-90.0, roll=0.0, yaw=0.0),
    )
    return world.spawn_actor(bp, transform, attach_to=vehicle)


# ---------------------------------------------------------------------------
# Live obstacle snapshot helpers
# ---------------------------------------------------------------------------

def _actor_live_poses(scenario, obstacles_gmm_base):
    """Return gmm entries with edge_point replaced by each actor's CURRENT
    world-frame position, queried live from CARLA.

    Works for both pedestrians and opposite vehicles: any sub-scenario that
    populated gmm_candidate_futures also put its actors into other_actors in
    the same order. We match them positionally rather than filtering by type,
    so the same code handles walkers ('walker' in type_id) and vehicles.
    """
    if not obstacles_gmm_base:
        return obstacles_gmm_base

    # Collect all dynamic actors across sub-scenarios in gmm_candidate_futures order.
    # save_gmm_label extends gmm from sub.gmm_candidate_futures in list_scenarios order;
    # other_actors is built the same way, so indices align.
    actor_list = []
    for sub in getattr(scenario, 'list_scenarios', []):
        if not getattr(sub, 'gmm_candidate_futures', None):
            continue   # sub has no GMM entries — skip its actors entirely
        for actor in getattr(sub, 'other_actors', []):
            try:
                if actor.is_alive:
                    actor_list.append(actor)
            except Exception:
                pass

    if not actor_list:
        return obstacles_gmm_base

    live = []
    for i, obs in enumerate(obstacles_gmm_base):
        new_obs = dict(obs)
        if i < len(actor_list):
            try:
                loc = actor_list[i].get_location()
                new_obs['edge_point'] = [loc.x, loc.y]
            except Exception:
                pass  # actor destroyed mid-episode; keep static pose
        live.append(new_obs)
    return live


def _patch_measurement_with_walkers(meas_path, walkers_live):
    """Append walkers_live into an already-written measurement JSON.gz in place."""
    with gzip.open(meas_path, 'rt', encoding='utf-8') as f:
        meas = ujson.load(f)
    meas['walkers_live'] = walkers_live
    with gzip.open(meas_path, 'wt', encoding='utf-8') as f:
        ujson.dump(meas, f)


# ---------------------------------------------------------------------------
# Heatmap post-processing
# ---------------------------------------------------------------------------

def _compute_heatmaps(save_path, obstacles_gmm_base, n_frames):
    """Post-process: render heatmap PNGs from saved measurements, then bake the
    model input into rgb/.

    For each frame we prefer the per-frame 'walkers_live' snapshot saved by
    on_step() (that JSON key is historical and also holds vehicles); if absent we
    fall back to obstacles_gmm_base (static world positions) so old episodes work.
    Frames with no obstacles get a blank (all-zero) heatmap and are still baked,
    so every frame ends up in the same stitched [camera | topdown+heatmap] format.
    """
    hm_dir = save_path / 'heatmap'
    print(f"  Computing {n_frames} heatmaps...", end="", flush=True)

    for i in range(n_frames):
        meas_path = save_path / 'measurements' / f'{i:04d}.json.gz'
        if not meas_path.exists():
            continue

        with gzip.open(meas_path, 'rt') as f:
            m = ujson.load(f)

        # Prefer live per-frame obstacle positions; fall back to static GMM.
        obstacles = m.get('walkers_live') or obstacles_gmm_base
        mat    = np.array(m['ego_matrix'])
        ego_xy = mat[:2, 3]
        yaw    = float(np.arctan2(mat[1, 0], mat[0, 0]))

        # Static parked-car boxes (ego frame) for this frame, so the heatmap
        # curves around them. Shared with heatmap_agent (built live there).
        static_boxes = None
        if STATIC_OBSTACLES_ENABLED:
            box_path = save_path / 'boxes' / f'{i:04d}.json.gz'
            if box_path.exists():
                with gzip.open(box_path, 'rt') as bf:
                    static_boxes = [b for b in ujson.load(bf)
                                    if b.get('class') == 'car']

        # Shared with heatmap_agent._render_risk so offline == live heatmaps.
        arr = render_route_risk_grid(
            obstacles, ego_xy, yaw, m.get('route') or None,
            TOPDOWN_SIZE, TOPDOWN_HALF, ROUTE_CORRIDOR_HALF_WIDTH,
            resolution=TOPDOWN_RES, horizon_scale=HEATMAP_HORIZON,
            window='cone', ego_speed=float(m.get('speed', 0.0)),
            static_boxes=static_boxes)
        Image.fromarray(arr).save(hm_dir / f'{i:04d}.png')

        # COLLISION-LOSS: un-masked obstacle risk grid for the training collision
        # loss (route_pts=None + default window => no cone/corridor clip). Sampled
        # under the predicted waypoints in DrivingAdaptor.compute_loss. DYNAMIC
        # obstacles only -- no static_boxes here, so parked cars don't carve holes
        # into (or otherwise touch) the loss teacher; the loss is dynamic-only.
        risk_full = render_route_risk_grid(
            obstacles, ego_xy, yaw, None,
            RISK_GRID_SIZE, TOPDOWN_HALF, ROUTE_CORRIDOR_HALF_WIDTH,
            resolution=TOPDOWN_RES, horizon_scale=HEATMAP_HORIZON)
        (save_path / 'risk_grid').mkdir(exist_ok=True)
        Image.fromarray(risk_full).save(save_path / 'risk_grid' / f'{i:04d}.png')

        # Save the topdown collision-heatmap view (BEV camera + colorized risk
        # overlay) that SimLingo consumes as a SECOND image tile alongside the
        # plain front camera. rgb/ is left untouched (plain front camera).
        topdown_path = save_path / 'topdown' / f'{i:04d}.jpg'
        if topdown_path.exists():
            topdown_arr = np.array(Image.open(topdown_path).convert('RGB'))
            overlay = heatmap_on_topdown(topdown_arr, arr)
            Image.fromarray(overlay).save(save_path / 'topdown_heatmap' / f'{i:04d}.jpg', quality=90)

    print(" done")

    # heatmap/ (grayscale) is a cheap-to-recompute intermediate -- always drop it.
    shutil.rmtree(hm_dir, ignore_errors=True)
    # topdown/ (clean BEV) is the only NON-regenerable overlay input. Keep it so
    # rebake.py can rebuild topdown_heatmap/ offline for any heatmap change (no
    # re-collection); drop it only if rebake-friendly storage is disabled.
    if not KEEP_TOPDOWN_BASE:
        shutil.rmtree(save_path / 'topdown', ignore_errors=True)


# ---------------------------------------------------------------------------
# Episode collection
# ---------------------------------------------------------------------------

def collect_episode(world, save_path, episode_type, destination, parked_spots):
    save_path = pathlib.Path(save_path)
    if save_path.exists():
        shutil.rmtree(save_path)
    for d in ('rgb', 'rgb_augmented', 'topdown', 'heatmap', 'topdown_heatmap',
              'measurements', 'boxes'):
        (save_path / d).mkdir(parents=True)

    y_offset, x_offset = get_offsets(episode_type)
    scenario_mode = _EPISODE_MODE_MAP[episode_type]

    aug_translation  = float(np.random.uniform(-1.5, 1.5))
    aug_rotation_deg = float(np.random.uniform(-20.0, 20.0))

    topdown_q        = queue.Queue(maxsize=2)
    frame_q          = queue.Queue(maxsize=2)
    aug_frame_q      = queue.Queue(maxsize=2)
    topdown_cam      = [None]
    camera           = [None]
    aug_camera       = [None]
    frame_idx        = [0]
    tick_count       = [0]
    obstacles_gmm_base = [None]

    def on_scenario_ready(scenario):
        topdown_cam[0] = setup_topdown_camera(world, scenario.car.actor)
        topdown_cam[0].listen(
            lambda img: topdown_q.put(img) if not topdown_q.full() else None
        )
        camera[0] = setup_camera(world, scenario.car.actor)
        camera[0].listen(lambda img: frame_q.put(img) if not frame_q.full() else None)
        aug_camera[0] = setup_camera(
            world, scenario.car.actor,
            y_offset=aug_translation, yaw_offset=aug_rotation_deg,
        )
        aug_camera[0].listen(lambda img: aug_frame_q.put(img) if not aug_frame_q.full() else None)
        save_gmm_label(save_path, scenario)
        gmm_path = save_path / "gmm.json"
        if gmm_path.exists():
            with open(gmm_path) as _f:
                obstacles_gmm_base[0] = json.load(_f).get("walkers", [])

    def _save_bgra(img, out):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(
            (img.height, img.width, 4))
        Image.fromarray(arr[:, :, :3][..., ::-1]).save(out, quality=95)

    def on_step(scenario):
        tick_count[0] += 1
        if tick_count[0] % SAVE_EVERY_N != 0:
            return
        if topdown_q.empty() or frame_q.empty():
            return

        idx = frame_idx[0]
        _save_bgra(topdown_q.get(), save_path / 'topdown' / f'{idx:04d}.jpg')
        _save_bgra(frame_q.get(),   save_path / 'rgb'     / f'{idx:04d}.jpg')
        if not aug_frame_q.empty():
            _save_bgra(aug_frame_q.get(), save_path / 'rgb_augmented' / f'{idx:04d}.jpg')
        save_measurement(save_path, idx, scenario, destination,
                         aug_translation=aug_translation, aug_rotation=aug_rotation_deg)
        save_boxes(save_path, idx, scenario)

        if obstacles_gmm_base[0]:
            live = _actor_live_poses(scenario, obstacles_gmm_base[0])
            if live:
                meas_path = save_path / 'measurements' / f'{idx:04d}.json.gz'
                _patch_measurement_with_walkers(meas_path, live)

        frame_idx[0] += 1

    ious, weighted_ious = [], []
    collisions_ref, near_misses_ref, walker_collisions_ref, actual_collisions = \
        [0], [0], [0], []

    try:
        run_scenario(
            world, destination, parked_spots,
            ious, weighted_ious, collisions_ref, near_misses_ref,
            actual_collisions, walker_collisions_ref,
            recording_path=None, car_list=[None],
            scenario_mode=scenario_mode,
            start_y_offset=y_offset, start_x_offset=x_offset,
            on_step=on_step, on_scenario_ready=on_scenario_ready,
        )
    finally:
        for cam in (topdown_cam[0], camera[0], aug_camera[0]):
            if cam is not None:
                cam.stop()
                cam.destroy()

    success = (
        len(ious) > 0 and ious[0] > 0.7 and
        collisions_ref[0] == 0 and walker_collisions_ref[0] == 0
    )

    if success:
        _compute_heatmaps(save_path, obstacles_gmm_base[0], frame_idx[0])

        results = {
            'status': 'completed',
            'scores': {'score_composed': 100.0, 'score_route': 100.0},
            'num_infractions': 0,
            'infractions': {
                'min_speed_infractions': [],
                'outside_route_lanes': [],
            },
        }
        with gzip.open(save_path / 'results.json.gz', 'wt', encoding='utf-8') as f:
            ujson.dump(results, f)

        meta = {
            'episode_type':       episode_type,
            'destination':        destination,
            'y_offset':           y_offset,
            'x_offset':           x_offset,
            'iou':                ious[0] if ious else None,
            'vehicle_collisions': collisions_ref[0],
            'walker_collisions':  walker_collisions_ref[0],
            'frames_saved':       frame_idx[0],
            'topdown':            {
                'height': TOPDOWN_HEIGHT,
                'size':   TOPDOWN_SIZE,
                'fov':    TOPDOWN_FOV,
            },
        }
        with open(save_path / 'episode_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
    else:
        shutil.rmtree(save_path)

    iou_str = f"{ious[0]:.2f}" if ious else "n/a"
    print(f"  Episode {'SUCCESS' if success else 'FAILED'} "
          f"— {frame_idx[0]} frames, iou={iou_str}")
    return success


# ---------------------------------------------------------------------------
# Dataset loop
# ---------------------------------------------------------------------------

def collect_dataset(world, output_dir):
    episode_types = (
        ['normal']               * 50  +
        ['normal_empty']         * 50  +
        ['normal_close']         * 50  +
        ['normal_close_empty']   * 50  +
        ['recovery']             * 50  +
        ['pedestrian_normal']    * 150 +
        ['pedestrian_recovery']  * 50  +
        ['door_normal']          * 50  +
        ['opposite_collide']     * 50  +
        ['opposite_near_miss']   * 50  +
        ['opposite_stop_early']  * 50
    )
    random.shuffle(episode_types)

    _override = os.environ.get("COLLECT_EPISODES")
    if _override:
        episode_types = []
        for part in _override.split(","):
            etype, _, count = part.strip().partition(":")
            episode_types += [etype] * int(count or 1)
        print(f"[COLLECT_EPISODES override] "
              f"{len(episode_types)} episodes: {episode_types}")

    output_dir  = pathlib.Path(output_dir)
    succeeded, failed = 0, 0
    _episode_base = (output_dir / 'data' / 'simlingo' / 'parking_ft' /
                     'routes_training' / 'RouteScenario_parking')
    if _episode_base.exists():
        shutil.rmtree(_episode_base)
    _episode_base.mkdir(parents=True)

    for i, episode_type in enumerate(episode_types):
        destination, parked_spots = random.choice(TRAIN_SCENARIOS)
        save_path = _episode_base / f'Town04_{i:04d}'
        print(f"[{i+1}/{len(episode_types)}] {episode_type} → {save_path}")
        try:
            ok = collect_episode(world, save_path, episode_type,
                                 destination, parked_spots)
        except Exception as e:
            print(f"  Episode CRASHED: {e}")
            import traceback; traceback.print_exc()
            if save_path.exists():
                shutil.rmtree(save_path)
            ok = False
        succeeded += int(bool(ok))
        failed    += int(not ok)

    print(f"\nDone: {succeeded} succeeded, {failed} failed "
          f"out of {len(episode_types)} episodes")


if __name__ == '__main__':
    from testbed.v2_experiment_utils import (
        load_client, town04_load, town04_spectator_bev,
    )
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

    client = load_client()
    world  = town04_load(client)
    CarlaDataProvider.set_client(client)
    CarlaDataProvider.set_world(world)
    town04_spectator_bev(world)

    settings = world.get_settings()
    settings.fixed_delta_seconds = _FIXED_DT
    world.apply_settings(settings)

    collect_dataset(world, OUTPUT)