"""
map_process.py — Static parking lot lane geometry for DiffusionPlanner.

Builds the `lanes` tensor from hardcoded parking lot geometry.
Route lanes are built separately in diffusion_adapter.py from the goal waypoint.

Public API
----------
    build_map_features(anchor_ego_state) -> lanes  (LANE_NUM, LANE_LEN, 12)
"""

import numpy as np
from typing import Optional
import sys
sys.path.insert(0, '/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet')
sys.path.insert(0, '/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/diffusion_adapter')
sys.path.insert(0, '/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/diffusion_adapter/nuplan-devkit')

from utils.coord_utils import carla_transform_to_standard
from parking_position import parking_vehicle_locations_Town04

# ---------------------------------------------------------------------------
# Constants — must match DataProcessor / DiffusionPlanner config
# ---------------------------------------------------------------------------
LANE_NUM         = 70
ROUTE_LANE_NUM   = 25
LANE_LEN         = 20
LANE_WIDTH       = 3.5
LANE_FEATURE_DIM = 12

# ---------------------------------------------------------------------------
# Parking lot geometry (CARLA frame → converted to standard on init)
# ---------------------------------------------------------------------------
ROW2_X        = 290.9
ROW3_X        = 280.0
AISLE_23_X    = (ROW2_X + ROW3_X) / 2.0
AISLE_ENTRY_X = 285.6
AISLE_Y_START = -244.0
AISLE_Y_END   = -185.0
SPOT_DEPTH    = 5.5

ROW2_SPOT_YS = [loc.y for loc in parking_vehicle_locations_Town04[16:32]]
ROW3_SPOT_YS = [loc.y for loc in parking_vehicle_locations_Town04[32:48]]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _carla_polyline_to_standard(pts_carla: np.ndarray) -> np.ndarray:
    result = np.zeros_like(pts_carla, dtype=np.float64)
    for i, (cx, cy) in enumerate(pts_carla):
        sx, sy, _ = carla_transform_to_standard(cx, cy, 0.0)
        result[i] = [sx, sy]
    return result


def _make_aisle(x: float, y_start: float, y_end: float, n_pts: int = 20) -> np.ndarray:
    ys = np.linspace(y_start, y_end, n_pts)
    xs = np.full_like(ys, x)
    return _carla_polyline_to_standard(np.stack([xs, ys], axis=1))


def _make_branch(aisle_x: float, spot_y: float, row_x: float, n_pts: int = 8) -> np.ndarray:
    direction = 1.0 if row_x > aisle_x else -1.0
    xs = np.linspace(aisle_x, aisle_x + direction * SPOT_DEPTH, n_pts)
    ys = np.full_like(xs, spot_y)
    return _carla_polyline_to_standard(np.stack([xs, ys], axis=1))


def _interpolate_polyline(points: np.ndarray, num_points: int) -> np.ndarray:
    if len(points) < 2:
        return np.tile(points[0:1], (num_points, 1))
    diffs    = np.diff(points, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cumlen   = np.concatenate([[0], np.cumsum(seg_lens)])
    total    = cumlen[-1]
    if total < 1e-6:
        return np.tile(points[0:1], (num_points, 1))
    targets   = np.linspace(0, total, num_points)
    resampled = np.zeros((num_points, 2), dtype=np.float64)
    for i, t in enumerate(targets):
        idx = np.clip(np.searchsorted(cumlen, t, side='right') - 1, 0, len(points) - 2)
        seg = seg_lens[idx]
        resampled[i] = points[idx] if seg < 1e-9 else \
            points[idx] + (t - cumlen[idx]) / seg * diffs[idx]
    return resampled


def _offset_polyline(center: np.ndarray, offset: float) -> np.ndarray:
    if abs(offset) < 1e-6:
        return center.copy()
    fwd   = np.vstack([np.diff(center, axis=0), np.diff(center, axis=0)[-1:]])
    norms = np.linalg.norm(fwd, axis=1, keepdims=True)
    fwd   = fwd / np.where(norms < 1e-6, 1.0, norms)
    return center + np.stack([-fwd[:, 1], fwd[:, 0]], axis=1) * offset


def _make_boundaries(center: np.ndarray, width: float = LANE_WIDTH):
    fwd   = np.vstack([np.diff(center, axis=0), np.diff(center, axis=0)[-1:]])
    norms = np.linalg.norm(fwd, axis=1, keepdims=True)
    fwd   = fwd / np.where(norms < 1e-6, 1.0, norms)
    left_normal = np.stack([-fwd[:, 1], fwd[:, 0]], axis=1)
    return center + left_normal * (width / 2), center - left_normal * (width / 2)


def _build_lane_feature(
    center: np.ndarray,            # (LANE_LEN, 2) in standard global frame
    left:   np.ndarray,            # (LANE_LEN, 2)
    right:  np.ndarray,            # (LANE_LEN, 2)
    anchor: np.ndarray,            # (3,) [x, y, heading] ego in standard frame
    traffic_light: np.ndarray = None,
) -> np.ndarray:
    """
    Build (LANE_LEN, 12) lane feature matching training _lane_polyline_process.
    Layout: [x, y, dx, dy, lx, ly, rx, ry, tl×4]  — all in ego-centric frame.
    """
    from diffusion_planner.data_process.utils import vector_set_coordinates_to_local_frame

    avails     = np.ones((1, LANE_LEN), dtype=np.bool_)
    center_ego = vector_set_coordinates_to_local_frame(center[np.newaxis], avails, anchor)[0]
    left_ego   = vector_set_coordinates_to_local_frame(left[np.newaxis],   avails, anchor)[0]
    right_ego  = vector_set_coordinates_to_local_frame(right[np.newaxis],  avails, anchor)[0]

    fwd      = np.vstack([np.diff(center_ego, axis=0), np.zeros((1, 2))])
    to_left  = left_ego  - center_ego
    to_right = right_ego - center_ego
    tl       = np.tile([0, 0, 0, 1], (LANE_LEN, 1)).astype(np.float32) \
               if traffic_light is None else traffic_light

    return np.concatenate([center_ego, fwd, to_left, to_right, tl], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Static lane list — built once at import time
# ---------------------------------------------------------------------------

def _build_static_lanes():
    lanes = []
    lanes.append(('aisle_23',    _make_aisle(AISLE_23_X,    AISLE_Y_START, AISLE_Y_END)))
    lanes.append(('aisle_entry', _make_aisle(AISLE_ENTRY_X, AISLE_Y_START, AISLE_Y_END)))
    return lanes

_STATIC_LANES = _build_static_lanes()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_map_features(anchor: np.ndarray):
    """
    Build the lanes tensor for DiffusionPlanner.
    Route lanes are NOT built here — they come from diffusion_adapter._make_straight_route_lanes().

    :param anchor: (3,) [std_x, std_y, std_heading] current ego pose in standard frame
    :return: lanes (LANE_NUM, LANE_LEN, 12)  — zero-padded if fewer lanes are nearby
    """
    lanes_out = np.zeros((LANE_NUM, LANE_LEN, LANE_FEATURE_DIM), dtype=np.float32)
    ego_xy    = anchor[:2]
    lane_idx  = 0

    for tag, polyline in _STATIC_LANES:
        if lane_idx >= LANE_NUM:
            break

        # Skip lanes more than 40m away
        if np.linalg.norm(polyline - ego_xy, axis=1).min() > 40.0:
            continue

        resampled    = _interpolate_polyline(polyline, LANE_LEN)
        left, right  = _make_boundaries(resampled)
        lanes_out[lane_idx] = _build_lane_feature(resampled, left, right, anchor)
        lane_idx += 1

        # Lateral copies for aisles
        if tag in ('aisle_23', 'aisle_entry'):
            for offset in [-LANE_WIDTH, LANE_WIDTH]:
                if lane_idx >= LANE_NUM:
                    break
                shifted     = _offset_polyline(resampled, offset)
                left, right = _make_boundaries(shifted)
                lanes_out[lane_idx] = _build_lane_feature(shifted, left, right, anchor)
                lane_idx += 1

    print(f"[map_process] {lane_idx} lanes built for anchor ({ego_xy[0]:.1f},{ego_xy[1]:.1f})")
    return lanes_out