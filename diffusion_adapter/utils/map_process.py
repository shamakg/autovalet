'''
A* path is computed once at startup
'''

import numpy as np
from typing import List, Optional
 
from utils.coord_utils import carla_transform_to_standard
from parking_position import parking_vehicle_locations_Town04

# ---------------------------------------------------------------------------
# Constants matching DataProcessor (this is built into the diffusion model)
# ---------------------------------------------------------------------------
LANE_NUM        = 70
ROUTE_LANE_NUM  = 25
LANE_LEN        = 20    # points per polyline
LANE_WIDTH      = 3.5   # [m] assumed lane width for boundary estimation
LANE_FEATURE_DIM = 12

######### Don't touch the above
 
TL_UNKNOWN = np.array([0, 0, 0, 1], dtype=np.float32)
 
# Lateral offsets to generate parallel lanes from the A* path
# These fill the lanes tensor with plausible nearby lanes
LANE_OFFSETS = [0.0, -LANE_WIDTH, LANE_WIDTH, -2*LANE_WIDTH, 2*LANE_WIDTH]

######## SETTING UP LANES FOR THE MAP
ROW2_X = 290.9
ROW3_X = 280.0

AISLE_23_X = (ROW2_X + ROW3_X) / 2.0  # getting the lane width
AISLE_ENTRY_X = 285.6
AISLE_Y_START = -244.0
AISLE_Y_END   = -185.0

SPOT_DEPTH = 5.5 #m
## get loc of each y position
ROW2_SPOT_YS = [loc.y for loc in parking_vehicle_locations_Town04[16:32]]
ROW3_SPOT_YS = [loc.y for loc in parking_vehicle_locations_Town04[32:48]]
###########

def _carla_polyline_to_standard(pts_carla):
    '''go from carla to standard frame'''
    result = np.zeros_like(pts_carla, dtype=np.float64)
    for i, (cx, cy) in enumerate(pts_carla):
        sx, sy, _ = carla_transform_to_standard(cx, cy, 0.0)
        result[i] = [sx, sy]
    return result

def make_aisle(x, y_start, y_end, n_pts = 20):
    ys = np.linspace(y_start, y_end, n_pts)
    xs = np.full_like(ys, x)
    return _carla_polyline_to_standard(np.stack([xs, ys], axis=1))

def make_branch(aisle_x, spot_y, row_x, n_pts = 8):
    direction = 1.0 if row_x > aisle_x else -1.0 ## up or down
    xs = np.linspace(aisle_x, aisle_x + direction * SPOT_DEPTH, n_pts)
    ys = np.full_like(xs, spot_y)
    return _carla_polyline_to_standard(np.stack([xs, ys], axis=1))

def _build_static_lanes():
    lanes = []
 
    # Main aisle between rows 2 and 3
    lanes.append(('aisle_23', make_aisle(AISLE_23_X, AISLE_Y_START, AISLE_Y_END)))
 
    # Entry aisle (from spawn point)
    lanes.append(('aisle_entry', make_aisle(AISLE_ENTRY_X, AISLE_Y_START, AISLE_Y_END)))
 
    # Branch lanes into each row 2 spot
    for spot_y in ROW2_SPOT_YS:
        lanes.append(('branch_row2', make_branch(AISLE_23_X, spot_y, ROW2_X)))
 
    # Branch lanes into each row 3 spot
    for spot_y in ROW3_SPOT_YS:
        lanes.append(('branch_row3', make_branch(AISLE_23_X, spot_y, ROW3_X)))
 
    return lanes

_STATIC_LANES = _build_static_lanes()



def _interpolate_polyline(points: np.ndarray, num_points: int) -> np.ndarray:
    """
    Resample a polyline to exactly num_points via linear interpolation.
 
    :param points: (N, 2) array of [x, y]
    :param num_points: desired output length
    :return: (num_points, 2)
    """
    if len(points) < 2:
        return np.tile(points[0:1], (num_points, 1))
 
    diffs      = np.diff(points, axis=0)
    seg_lens   = np.linalg.norm(diffs, axis=1)
    cumlen     = np.concatenate([[0], np.cumsum(seg_lens)])
    total_len  = cumlen[-1]
 
    if total_len < 1e-6:
        return np.tile(points[0:1], (num_points, 1))
 
    targets    = np.linspace(0, total_len, num_points)
    resampled  = np.zeros((num_points, 2), dtype=np.float64)
 
    for i, t in enumerate(targets):
        idx      = np.searchsorted(cumlen, t, side='right') - 1
        idx      = np.clip(idx, 0, len(points) - 2)
        seg_len  = seg_lens[idx]
        if seg_len < 1e-9:
            resampled[i] = points[idx]
        else:
            alpha        = (t - cumlen[idx]) / seg_len
            resampled[i] = points[idx] + alpha * diffs[idx]
 
    return resampled


def _offset_polyline(center: np.ndarray, offset: float) -> np.ndarray:
    """
    Shift a centerline laterally by offset metres.
    Positive offset = left, negative = right (standard frame convention).
 
    :param center: (N, 2)
    :param offset: lateral offset in metres
    :return: (N, 2)
    """
    if abs(offset) < 1e-6:
        return center.copy()
 
    fwd        = np.diff(center, axis=0)
    fwd        = np.vstack([fwd, fwd[-1:]])          # repeat last
    norms      = np.linalg.norm(fwd, axis=1, keepdims=True)
    norms      = np.where(norms < 1e-6, 1.0, norms)
    fwd        = fwd / norms
 
    # Left normal: rotate fwd 90° CCW
    left_norm  = np.stack([-fwd[:, 1], fwd[:, 0]], axis=1)
    return center + offset * left_norm

def _build_lane_feature(
    center: np.ndarray,
    anchor_ego_state: np.ndarray,
) -> np.ndarray:
    """
    Build a (LANE_LEN, 12) lane feature array in ego-centric frame.
 
    :param center: (LANE_LEN, 2) centerline in standard global frame
    :param anchor_ego_state: (3,) [x, y, heading] of ego in standard frame
    :return: (LANE_LEN, 12)
    """
    from diffusion_planner.data_process.utils import vector_set_coordinates_to_local_frame
 
    resampled  = _interpolate_polyline(center, LANE_LEN)
    N = resampled.shape[0]
 
    # Transform centerline to ego frame
    avails         = np.ones((1, N), dtype=np.bool_)
    center_3d      = resampled[np.newaxis, :, :]          # (1, N, 2)
    center_ego     = vector_set_coordinates_to_local_frame(
        center_3d, avails, anchor_ego_state
    )[0]                                               # (N, 2)
 
    # Forward vector (match training: last point gets zero vector)
    fwd            = np.diff(center_ego, axis=0)
    fwd            = np.vstack([fwd, np.zeros((1, 2))])  # (N, 2)
 
    # Boundary offsets (in ego frame, just lateral)
    left_norm      = np.stack([-fwd[:, 1], fwd[:, 0]], axis=1)
    norms          = np.linalg.norm(left_norm, axis=1, keepdims=True)
    norms          = np.where(norms < 1e-6, 1.0, norms)
    left_norm      = left_norm / norms
 
    half_w         = LANE_WIDTH / 2.0
    to_left        = left_norm  *  half_w              # (N, 2)
    to_right       = left_norm  * -half_w              # (N, 2)
 
    # Traffic light: unknown for all
    tl             = np.tile(TL_UNKNOWN, (N, 1))       # (N, 4)
 
    feature = np.concatenate([
        center_ego,    # (N, 2)  x, y
        fwd,           # (N, 2)  dx, dy
        to_left,       # (N, 2)  lx, ly
        to_right,      # (N, 2)  rx, ry
        tl,            # (N, 4)
    ], axis=1).astype(np.float32)                      # (N, 12)
 
    return feature


class MapProcessor:
    """
    Precomputes the A* path once and serves ego-centric lane tensors each tick.
    """
 
    def __init__(self):
        # Full A* path in standard global frame: (M, 2)
        self._global_path: Optional[np.ndarray] = None
 
    def set_astar_path(self, trajectory):
        """
        Call once at startup with the Hybrid A* TrajectoryPoint list.
        Converts from CARLA frame to standard frame and caches.
 
        :param trajectory: List[TrajectoryPoint] in CARLA frame
        """
        if not trajectory:
            self._global_path = None
            return
 
        pts = []
        for tp in trajectory:
            std_x, std_y, _ = carla_transform_to_standard(tp.x, tp.y, 0.0)
            pts.append([std_x, std_y])
 
        self._global_path = np.array(pts, dtype=np.float64)  # (M, 2)
        print(f"[map_process] A* path cached: {len(pts)} points")
 
    def _extract_segments(self, closest: int, path: np.ndarray):
        """
        Extract multiple overlapping segments along the path, centred around
        different starting positions.  Each segment covers LANE_LEN points
        after interpolation.

        Returns a list of (LANE_LEN, 2) arrays in standard global frame.
        """
        N = len(path)
        segments = []

        # # Include some path behind the ego (up to 30 points back)
        behind_start = max(0, closest - 30)
        behind_end   = closest + 2  # at least 2 points
        if behind_end - behind_start >= 2:
            seg = _interpolate_polyline(path[behind_start:behind_end], LANE_LEN)
            segments.append(seg)

        # Forward segments at staggered starts along the path
        # Each segment covers a different portion of the upcoming route
        step = 10  # stride between segment starts (in path indices)
        for start_offset in range(0, 100, step):
            seg_start = closest + start_offset
            seg_end   = min(N, seg_start + 20)  # ~80 raw points per segment
            if seg_start >= N - 1:
                break
            if seg_end - seg_start < 2:
                break
            seg = _interpolate_polyline(path[seg_start:seg_end], LANE_LEN)
            segments.append(seg)

        return segments

    def build_map_features(
        self,
        anchor_ego_state: np.ndarray,
    ):
        """
        Build lanes and route_lanes tensors for DiffusionPlanner.
        Called every tick with the current ego pose.

        :param anchor_ego_state: (3,) [x, y, heading] in standard frame
        :return: (lanes, route_lanes)
            lanes:       np.ndarray (LANE_NUM, LANE_LEN, 12)
            route_lanes: np.ndarray (ROUTE_LANE_NUM, LANE_LEN, 12)
        """
        lanes_output       = np.zeros((LANE_NUM,       LANE_LEN, LANE_FEATURE_DIM), dtype=np.float32)
        route_lanes_output = np.zeros((ROUTE_LANE_NUM, LANE_LEN, LANE_FEATURE_DIM), dtype=np.float32)

        ego_xy  = anchor_ego_state[:2]


        # ------------------------------------------------------------------
        # 1. lanes — static parking lot geometry
        #    Distance filter: skip lanes more than 40m from ego
        # ------------------------------------------------------------------
        lane_idx = 0
        for tag, polyline in _STATIC_LANES:
            if lane_idx >= LANE_NUM:
                break
 
            dists = np.linalg.norm(polyline - ego_xy, axis=1)
            if dists.min() > 40.0:
                continue
 
            lanes_output[lane_idx] = _build_lane_feature(polyline, anchor_ego_state)
            lane_idx += 1
 
            # Give aisles lateral offset copies for lane-width context
            if tag in ('aisle_23', 'aisle_entry'):
                for offset in [-LANE_WIDTH, LANE_WIDTH]:
                    if lane_idx >= LANE_NUM:
                        break
                    shifted = _offset_polyline(polyline, offset)
                    lanes_output[lane_idx] = _build_lane_feature(shifted, anchor_ego_state)
                    lane_idx += 1
                    
        if self._global_path is None or len(self._global_path) < 2:
            return lanes_output, route_lanes_output

        ## build route_lanes (just use the A* path)
        if self._global_path is not None and len(self._global_path) >= 2:
            path    = self._global_path
            dists   = np.linalg.norm(path - ego_xy, axis=1)
            closest = int(np.argmin(dists))
 
            route_idx = 0
            for seg in self._extract_segments(closest, path):
                for offset in [0.0, -1.0, 1.0]:
                    if route_idx >= ROUTE_LANE_NUM:
                        break
                    shifted = _offset_polyline(seg, offset)
                    route_lanes_output[route_idx] = _build_lane_feature(shifted, anchor_ego_state)
                    route_idx += 1
                if route_idx >= ROUTE_LANE_NUM:
                    break   

        return lanes_output, route_lanes_output
 

_map_processor = MapProcessor()

def set_astar_path(trajectory):
    """
    Call once at startup with the Hybrid A* path.
    :param trajectory: List[TrajectoryPoint] from plan_hybrid_a_star()
    """
    _map_processor.set_astar_path(trajectory)

def build_map_features(anchor_ego_state: np.ndarray):
    """
    Call every tick to get lane tensors.
    :param anchor_ego_state: (3,) [x, y, heading] in standard frame
    :return: (lanes, route_lanes)
    """
    return _map_processor.build_map_features(anchor_ego_state)

def get_dist(anchor_ego_state):
    if _map_processor._global_path is None or len(_map_processor._global_path) < 2:
        return 0.0
    
    ego_xy = anchor_ego_state[:2]
    path = _map_processor._global_path

    dists = np.linalg.norm(path - ego_xy, axis=1)
    closest = int(np.argmin(dists))

    if closest == 0:
        return 0.0

    segs = np.linalg.norm(np.diff(path[:closest + 1], axis=0), axis=1)
    return float(np.sum(segs))


