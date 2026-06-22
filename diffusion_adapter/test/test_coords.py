"""
test_coords.py  —  Phase 1: coordinate transform sanity checks.

Run with:
    python test_coords.py

No CARLA, no model, no GPU needed.
"""

import sys, os, torch, numpy as np
import diff_adapter as da
_HERE = os.path.dirname(os.path.abspath(__file__))
_AUTOVALET_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _AUTOVALET_ROOT)
sys.path.insert(0, "/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/diffusion_adapter/nuplan-devkit")

_DP_ROOT = os.path.join(_HERE, "Diffusion-Planner")
if _DP_ROOT not in sys.path:
    sys.path.insert(0, _DP_ROOT)
"""
test_direction.py — Verify model output points toward destination.

This is the test we should have written before running in CARLA.
It checks that when given a destination to the right, the predicted
trajectory curves right, and vice versa.

Run with:
    python test_direction.py
"""

import sys, os
import numpy as np
import torch

sys.path.insert(0, '.')
sys.path.insert(0, 'Diffusion-Planner')
sys.path.insert(0, 'nuplan-devkit')

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from diffusion_planner.data_process.utils import (
    convert_to_model_inputs,
    _state_se2_array_to_transform_matrix,
    _state_se2_array_to_transform_matrix_batch,
    _transform_matrix_to_state_se2_array_batch,
)
from utils.coord_utils import carla_transform_to_standard, standard_to_carla
from utils.map_process import LANE_NUM

CKPT = 'checkpoints/model.pth'
ARGS = 'checkpoints/args.json'

def load():
    config = Config(ARGS, guidance_fn=da._guidance_fn)
    model  = Diffusion_Planner(config).cuda().eval()
    raw    = torch.load(CKPT, map_location='cuda')
    sd     = raw.get('ema_state_dict') or raw.get('model') or raw
    sd     = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model, config

def run(model, config, anchor, dest_std):
    """Run model with straight-line route from anchor to dest_std."""
    da._dest_std = dest_std
    da._config   = config
    da._config   = config

    M = _state_se2_array_to_transform_matrix(anchor)
    da._anchor_inv = np.linalg.inv(M)

    route_lanes = da._make_straight_route_lanes(anchor)

    data = {
        'neighbor_agents_past':        np.zeros((32,21,11), dtype=np.float32),
        'ego_current_state':           np.array([0,0,1,0,2,0,0,0,0,0], dtype=np.float32),
        'static_objects':              np.zeros((5,10), dtype=np.float32),
        'lanes':                       da.build_map_features(anchor),
        'lanes_speed_limit':           np.full((LANE_NUM,1), 14., dtype=np.float32),
        'lanes_has_speed_limit':       np.ones((LANE_NUM,1), dtype=np.bool_),
        'route_lanes':                 route_lanes,
        'route_lanes_speed_limit':     np.full((config.route_num,1), 14., dtype=np.float32),
        'route_lanes_has_speed_limit': np.ones((config.route_num,1), dtype=np.bool_),
    }
    inputs = convert_to_model_inputs(data, 'cuda')
    if config.observation_normalizer:
        inputs = config.observation_normalizer(inputs)
    with torch.no_grad():
        _, out = model(inputs)
    return out['prediction'][0, 0].cpu().numpy()  # (T, 4) ego-centric


def check(cond, tag, detail=''):
    status = '[PASS]' if cond else '[FAIL]'
    print(f"  {status} {tag}  {detail}")
    return cond


def test_direction(model, config):
    """
    Ego at spawn (285.6, -243.7, yaw=90°).
    Test 3 destinations:
      A) straight ahead  → trajectory should have near-zero lateral (y≈0)
      B) to the right    → trajectory should curve right (y < 0 in ego frame)
      C) to the left     → trajectory should curve left  (y > 0 in ego frame)
    """
    print('\n=== Direction test ===')
    print('Ego: CARLA (285.6, -243.7, yaw=90°)')

    # Anchor in nuPlan/standard frame (identity conversion now)
    anchor = np.array(carla_transform_to_standard(285.6, -243.7, 90.0))
    print(f'Anchor std: {anchor}')

    # In CARLA yaw=90° means facing +Y direction
    # Forward = +Y in CARLA = +Y in standard
    # Right   = +X in CARLA = +X in standard  
    # Left    = -X in CARLA = -X in standard

    cases = {
        'straight ahead': np.array([285.6, -183.7]),   # +60m in Y (forward)
        'right (+X)':     np.array([295.6, -243.7]),   # +10m in X (right)
        'left  (-X)':     np.array([275.6, -243.7]),   # -10m in X (left)
    }

    all_ok = True
    for name, dest_carla in cases.items():
        dest_std = np.array(carla_transform_to_standard(dest_carla[0], dest_carla[1], 0.0)[:2])
        pred = run(model, config, anchor, dest_std)

        # Final ego-frame position
        final_x, final_y = pred[-1, :2]
        # Mean lateral displacement over horizon
        mean_y = pred[:, 1].mean()

        print(f'\n  Destination: {name} CARLA=({dest_carla[0]:.1f},{dest_carla[1]:.1f})')
        print(f'  dest std: ({dest_std[0]:.1f},{dest_std[1]:.1f})')
        print(f'  final ego-frame: ({final_x:.2f}, {final_y:.2f})')
        print(f'  mean lateral y:  {mean_y:.3f}')

        if name == 'straight ahead':
            ok = check(abs(mean_y) < 3.0, 'straight → |mean_y| < 3m',
                      f'got {mean_y:.2f}')
        elif 'right' in name:
            ok = check(mean_y < 0, 'right dest → mean_y < 0 (curves right)',
                      f'got {mean_y:.2f}')
        elif 'left' in name:
            ok = check(mean_y > 0, 'left dest → mean_y > 0 (curves left)',
                      f'got {mean_y:.2f}')
        all_ok &= ok

    return all_ok


def test_output_in_carla_frame(model, config):
    """
    Convert model output back to CARLA and verify waypoints
    are on the correct side of the car.
    """
    print('\n=== Output CARLA frame test ===')

    anchor = np.array(carla_transform_to_standard(285.6, -243.7, 90.0))
    # Destination: 10m to the right in CARLA (+X)
    dest_std = np.array(carla_transform_to_standard(295.6, -243.7, 0.0)[:2])

    pred = run(model, config, anchor, dest_std)
    T = pred.shape[0]

    # Recover headings
    cos_h = pred[:, 2]; sin_h = pred[:, 3]
    norm  = np.sqrt(cos_h**2 + sin_h**2)
    norm  = np.where(norm < 1e-6, 1.0, norm)
    headings = np.arctan2(sin_h/norm, cos_h/norm)
    local_poses = np.stack([pred[:,0], pred[:,1], headings], axis=1)

    anchor_mat   = _state_se2_array_to_transform_matrix(anchor)
    pose_mats    = _state_se2_array_to_transform_matrix_batch(local_poses)
    global_poses = _transform_matrix_to_state_se2_array_batch(anchor_mat @ pose_mats)

    print(f'Ego CARLA: (285.6, -243.7)')
    print(f'Dest CARLA: (295.6, -243.7)  [10m to the RIGHT / +X]')
    print(f'wp[0]  CARLA: ({standard_to_carla(*global_poses[0])[0]:.2f}, {standard_to_carla(*global_poses[0])[1]:.2f})')
    print(f'wp[10] CARLA: ({standard_to_carla(*global_poses[10])[0]:.2f}, {standard_to_carla(*global_poses[10])[1]:.2f})')
    print(f'wp[-1] CARLA: ({standard_to_carla(*global_poses[-1])[0]:.2f}, {standard_to_carla(*global_poses[-1])[1]:.2f})')

    # wp[10] should have x > 285.6 (moved right) and y ≈ -243.7 (not drifted far forward)
    wp10_carla = standard_to_carla(*global_poses[10])
    ok1 = check(wp10_carla[0] > 285.6, f'wp[10] moved right (x > 285.6), got x={wp10_carla[0]:.2f}')
    ok2 = check(abs(wp10_carla[1] - (-243.7)) < 15.0,
                f'wp[10] not too far forward, got y={wp10_carla[1]:.2f}')

    return ok1 and ok2


if __name__ == '__main__':
    print('Loading model...')
    model, config = load()
    print('Model loaded.')

    r1 = test_direction(model, config)
    r2 = test_output_in_carla_frame(model, config)

    print(f'\n{"="*50}')
    passed = sum([r1, r2])
    print(f'Results: {passed}/2 test groups passed')
    if passed == 2:
        print('Coordinate convention verified — safe to run in CARLA.')
    else:
        print('Direction test FAILED — do NOT run in CARLA until fixed.')