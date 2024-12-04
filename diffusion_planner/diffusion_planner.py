"""
diffusion_planner.py

Adapter layer between the autovalet ObstacleMap/TrajectoryPoint world and a
diffusion-based trajectory planner (e.g. Diffusion-Planner, ParkDiffusion).

Phase 1: Adapters only — no live CARLA, no existing files modified.
         diffusion_plan() raises NotImplementedError until a model is wired in.

To integrate a real model:
  1. Load weights in load_model()
  2. Fill in diffusion_plan() with the actual inference call
  3. Set USE_DIFFUSION = True in v2_diffusion.py
"""

from __future__ import annotations
from math import sqrt, cos, sin, atan2
from typing import List, Optional
import numpy as np
import sys
import os

# Allow importing from the parent autovalet directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from v2_diffusion import (
    TrajectoryPoint, ObstacleMap, Direction,
    refine_trajectory, MIN_SPEED, MAX_SPEED, TRAJECTORY_EXTENSION
)

# ---------------------------------------------------------------------------
# Model handle — filled in by load_model()
# ---------------------------------------------------------------------------
_model = None

def load_model(checkpoint_path: str):
    """
    Load diffusion model weights.  Call once at startup before any inference.

    Example (Diffusion-Planner):
        from planner import DiffusionPlanner
        global _model
        _model = DiffusionPlanner.from_pretrained(checkpoint_path)
        _model.eval()
    """
    global _model
    # TODO: replace with actual model loading
    raise NotImplementedError(
        "load_model() not implemented. "
        "Wire in your model here (e.g. Diffusion-Planner or ParkDiffusion)."
    )


# ---------------------------------------------------------------------------
# Input adapter
# ---------------------------------------------------------------------------

# Local BEV crop size around ego (metres)
LOCAL_BEV_HALF_EXTENT = 20.0
# Grid resolution of ObstacleMap (metres/cell)
OBS_RESOLUTION = 0.25


def obs_to_diffusion_input(
    cur: TrajectoryPoint,
    destination: TrajectoryPoint,
    obs: ObstacleMap,
) -> dict:
    """
    Convert autovalet state into the input dict expected by a diffusion planner.

    Returns a dict with:
      'bev'        : (H, W) float32 binary occupancy, ego-centred, North-up
      'ego_start'  : [0.0, 0.0, 0.0]   (always origin in ego frame)
      'ego_goal'   : [rel_x, rel_y, rel_yaw]  goal in ego frame
      'agents'     : (N, 4) float32  [rel_x, rel_y, vx, vy] per tracked object
    """
    # --- BEV occupancy crop (ego-relative, axis-aligned) ---
    probs = obs.probs()                         # shape (W_map, H_map), probability of occupancy for each grid cell
    cx, cy = obs.transform_coord(cur.x, cur.y)  # convert the ego position to grid coords

    ## create square region around the vehicle in which it will see obstacles
    half = int(LOCAL_BEV_HALF_EXTENT / OBS_RESOLUTION) 
    x0, x1 = cx - half, cx + half ## puts the car at the center of the box
    y0, y1 = cy - half, cy + half

    # create a bev obstacle view centered around the car
    bev = np.ones((2 * half, 2 * half), dtype=np.float32)
    mx0, mx1 = max(x0, 0), min(x1, probs.shape[0]) ## truncate it if at the bottom
    my0, my1 = max(y0, 0), min(y1, probs.shape[1])
    bev[mx0 - x0: mx1 - x0, my0 - y0: my1 - y0] = (
        probs[mx0:mx1, my0:my1] > 0.5
    ).astype(np.float32)

    # --- Goal in ego frame ---
    dx = destination.x - cur.x
    dy = destination.y - cur.y
    cos_h = cos(-cur.angle)
    sin_h = sin(-cur.angle)
    rel_x = cos_h * dx - sin_h * dy
    rel_y = sin_h * dx + cos_h * dy
    rel_yaw = _wrap_angle(destination.angle - cur.angle)

    # --- Dynamic agent state vectors from Kalman filter ---
    # dyn_obs_clusters[id] = (centroid, cluster_points, time)
    # dyn_obs_states[id]   = (state_mean [x, y, vx, vy], state_cov)
    agents = []
    for obj_id, (centroid, _, _) in obs.dyn_obs_clusters.items():
        if obj_id not in obs.dyn_obs_states:
            continue
        state_mean, _ = obs.dyn_obs_states[obj_id]   # [x, y, vx, vy]
        ax, ay, avx, avy = state_mean
        # convert to ego-relative position through some basic trig
        adx, ady = ax - cur.x, ay - cur.y
        rel_ax = cos_h * adx - sin_h * ady
        rel_ay = sin_h * adx + cos_h * ady
        agents.append([rel_ax, rel_ay, avx, avy])

    agents_arr = np.array(agents, dtype=np.float32) if agents else np.zeros((0, 4), dtype=np.float32)

    return {
        'bev':       bev,
        'ego_start': np.array([0.0, 0.0, 0.0], dtype=np.float32),
        'ego_goal':  np.array([rel_x, rel_y, rel_yaw], dtype=np.float32),
        'agents':    agents_arr,
    }


# ---------------------------------------------------------------------------
# Output adapter
# ---------------------------------------------------------------------------

def diffusion_output_to_trajectory(
    raw_waypoints: np.ndarray,   # (N, 2) or (N, 3): [rel_x, rel_y] or [rel_x, rel_y, rel_yaw]
    cur: TrajectoryPoint,
) -> List[TrajectoryPoint]:
    """
    Convert ego-relative diffusion output waypoints to a List[TrajectoryPoint]
    in world frame with speed profiles filled in by refine_trajectory().
    """
    if len(raw_waypoints) < 2:
        return []

    cos_h = cos(cur.angle)
    sin_h = sin(cur.angle)

    trajectory = []
    prev_x, prev_y = cur.x, cur.y

    for i, wp in enumerate(raw_waypoints):
        rel_x, rel_y = float(wp[0]), float(wp[1])

        # Rotate from ego frame to world frame
        world_x = cur.x + cos_h * rel_x - sin_h * rel_y
        world_y = cur.y + sin_h * rel_x + cos_h * rel_y

        # Infer heading from motion direction if not provided
        if raw_waypoints.shape[1] >= 3:
            angle = _wrap_angle(cur.angle + float(wp[2]))
        else:
            dx, dy = world_x - prev_x, world_y - prev_y
            angle = atan2(dy, dx) if (dx != 0 or dy != 0) else cur.angle

        trajectory.append(TrajectoryPoint(Direction.FORWARD, world_x, world_y, MIN_SPEED, angle))
        prev_x, prev_y = world_x, world_y

    if not trajectory:
        return []

    trajectory[0].speed = cur.speed
    trajectory[0].angle = cur.angle
    refine_trajectory(trajectory)   # assigns speeds + FORWARD/REVERSE directions in-place
    return trajectory


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def diffusion_plan(
    cur: TrajectoryPoint,
    destination: TrajectoryPoint,
    obs: ObstacleMap,
) -> List[TrajectoryPoint]:
    """
    Run diffusion model inference and return a trajectory.

    Raises NotImplementedError until load_model() has been called and
    the inference call below is implemented.
    """
    if _model is None:
        raise NotImplementedError(
            "Diffusion model not loaded. Call load_model(checkpoint_path) first."
        )

    model_input = obs_to_diffusion_input(cur, destination, obs)

    # TODO: replace with actual model inference, e.g.:
    #   raw_waypoints = _model.predict(
    #       bev=model_input['bev'],
    #       goal=model_input['ego_goal'],
    #       agents=model_input['agents'],
    #   )
    raise NotImplementedError(
        "diffusion_plan() inference not yet implemented. "
        "Replace this with your model's forward pass."
    )

    trajectory = diffusion_output_to_trajectory(raw_waypoints, cur)
    return trajectory


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle
