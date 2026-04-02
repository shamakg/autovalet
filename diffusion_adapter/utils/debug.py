import os
import torch
import numpy as np
from typing import List, Optional
import sys
import carla

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from diffusion_planner.model.guidance.collision import collision_guidance_fn
from diffusion_planner.data_process.utils import (
    convert_to_model_inputs,
    _state_se2_array_to_transform_matrix,
    _state_se2_array_to_transform_matrix_batch,
    _transform_matrix_to_state_se2_array_batch,
    _global_state_se2_array_to_local,
    _global_velocity_to_local,
)

from utils.coord_utils import (
    carla_transform_to_standard,
    carla_velocity_to_standard,
    standard_to_carla,
)
from utils.agent_process import (
    AgentHistoryBuffer,
    AgentState,
    NUM_AGENTS,
    NUM_PAST_STEPS,
)
from utils.guidance import _CombinedGuidance
from v2 import TrajectoryPoint, Direction, MIN_SPEED, MAX_SPEED, plan_hybrid_a_star
from utils.map_process import build_map_features, set_astar_path, LANE_NUM, ROUTE_LANE_NUM, get_dist

def _draw_route_lanes_debug(world, anchor_ego_state, route_lanes_tensor):
    route_lanes_np = route_lanes_tensor.cpu().numpy() if hasattr(route_lanes_tensor, 'cpu') else route_lanes_tensor
    
    for lane_idx in range(min(12, route_lanes_np.shape[0])):
        lane = route_lanes_np[lane_idx]  # (20, 12)
        
        # Skip empty lanes
        if np.abs(lane[:, :2]).sum() < 1e-6:
            continue
            
        for pt_idx in range(0, lane.shape[0], 2):  # every other point

            
            ego_x, ego_y = float(lane[pt_idx, 0]), float(lane[pt_idx, 1])
            
            h = anchor_ego_state[2]
            std_x = anchor_ego_state[0] + ego_x * np.cos(h) - ego_y * np.sin(h)
            std_y = anchor_ego_state[1] + ego_x * np.sin(h) + ego_y * np.cos(h)
            
            carla_x, carla_y, _ = standard_to_carla(std_x, std_y, 0.0)
            
            color = carla.Color(r=0, g=255, b=0) if lane_idx == 0 else carla.Color(r=255, g=165, b=0)
            world.debug.draw_point(
                carla.Location(x=carla_x, y=carla_y, z=0.5),
                size=0.1,
                color=color,
                life_time=1.0
            )


            color2 = carla.Color(r=200, g=35, b=0)
            if pt_idx == 0 and lane_idx == 0:
                world.debug.draw_string(
                    carla.Location(x=carla_x, y=carla_y, z=1.0),
                    f'R{lane_idx}',
                    color=color2,
                    life_time=0.1
                )

def _draw_ego_frame_debug(world, anchor_ego_state, dest_local, path_ego):
    """Draw destination and path in ego frame, transformed back to world for visualization."""
    x0, y0 = anchor_ego_state[0], anchor_ego_state[1]
    
    def ego_to_world_carla(ex, ey):
        wx, wy = ego_to_world(ex, ey, x0, y0, anchor_ego_state[2])
        cx, cy, _ = standard_to_carla(wx, wy, 0.0)
        return cx, cy
    
    dx, dy = ego_to_world_carla(dest_local[0], dest_local[1])
    # Draw destination (red) - very prominent
    
    loc = carla.Location(x=dx, y=dy, z=1.0)

    

    # # Tall vertical line
    # world.debug.draw_line(
    #     carla.Location(x=dx, y=dy, z=0.0),
    #     carla.Location(x=dx, y=dy, z=5.0),
    #     thickness=0.3,
    #     color=carla.Color(r=255, g=0, b=0),
    #     life_time=0.5
    # )

    
    
    print(f"GOAL drawn at CARLA: ({dx:.2f}, {dy:.2f}), ego frame: ({dest_local[0]:.2f}, {dest_local[1]:.2f})")

    # # Draw path points (blue)
    # for i in range(0, len(path_ego), 5):
    #     px, py = ego_to_world_carla(path_ego[i, 0], path_ego[i, 1])
    #     world.debug.draw_point(
    #         carla.Location(x=px, y=py, z=0.5),
    #         size=0.1,
    #         color=carla.Color(r=0, g=0, b=255),
    #         life_time=1.0
    #     )

    # Text label
    world.debug.draw_string(
        carla.Location(x=dx, y=dy, z=5.0),
        'GOAL',
        color=carla.Color(r=0, g=255, b=0),
        life_time=0.5
    )
    # Large red sphere
    world.debug.draw_point(
        loc,
        size=0.1,
        color=carla.Color(r=0, g=255, b=0),
        life_time=0.5
    )

def debug_ego_axes(world, anchor_ego_state, length=5.0):
    """Draw ego's +x (forward) and +y (left) axes in CARLA world so you can
    visually verify that agent positions have the right sign convention."""
    from utils.coord_utils import standard_to_carla
    x0, y0, h = anchor_ego_state

    def draw_axis(dx_ego, dy_ego, color, label):
        wx = x0 + np.cos(h) * dx_ego - np.sin(h) * dy_ego
        wy = y0 + np.sin(h) * dx_ego + np.cos(h) * dy_ego
        cx0, cy0, _ = standard_to_carla(x0, y0, 0.0)
        cx1, cy1, _ = standard_to_carla(wx, wy, 0.0)
        world.debug.draw_arrow(
            carla.Location(x=cx0, y=cy0, z=1.0),
            carla.Location(x=cx1, y=cy1, z=1.0),
            thickness=0.08, arrow_size=0.2,
            color=color, life_time=0.5
        )
        world.debug.draw_string(
            carla.Location(x=cx1, y=cy1, z=1.5),
            label, color=color, life_time=0.5
        )

    draw_axis(length, 0.0,   carla.Color(0, 255, 0),   '+x fwd')   # green = forward
    draw_axis(0.0,   length, carla.Color(0, 0, 255),   '+y left')  # blue  = left


def debug_agents(world, neighbor_np, x0, y0, h):
    ### DEBUG
    for slot_idx in range(neighbor_np.shape[0]):
        last_step = neighbor_np[slot_idx, -1]
        if np.abs(last_step).sum() < 1e-6:
            continue
        ex, ey = float(last_step[0]), float(last_step[1])
        cos_h, sin_h = float(last_step[2]), float(last_step[3])
        local_heading = np.arctan2(sin_h, cos_h)
        half_l = float(last_step[7]) / 2.0
        half_w = float(last_step[6]) / 2.0

        # Compute world heading directly
        world_heading = local_heading + h

        wx, wy = ego_to_world(ex, ey, x0, y0, h)
        cx, cy, _ = standard_to_carla(wx, wy, 0.0)
        dist = np.sqrt(ex**2 + ey**2)
        color = carla.Color(r=140, g=30, b=140) if dist < 10.0 else carla.Color(r=30, g=120, b=120)

        # Draw 4-corner bounding box rectangle in world frame
        # fwd = along heading, left = 90° CCW from heading
        cos_wh = np.cos(world_heading)
        sin_wh = np.sin(world_heading)
        # 4 corners: (±half_l along fwd) ± (half_w along left)
        corners_std = []
        for sl, sw in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
            corner_wx = wx + sl * half_l * cos_wh - sw * half_w * sin_wh
            corner_wy = wy + sl * half_l * sin_wh + sw * half_w * cos_wh
            ccx, ccy, _ = standard_to_carla(corner_wx, corner_wy, 0.0)
            corners_std.append(carla.Location(x=ccx, y=ccy, z=1.0))

        # Draw 4 edges
        for i in range(4):
            world.debug.draw_line(
                corners_std[i], corners_std[(i + 1) % 4],
                thickness=0.05, color=color, life_time=1.0
            )

        world.debug.draw_string(
            carla.Location(x=cx, y=cy, z=2.0),
            f'A{slot_idx} {dist:.1f}m',
            color=color,
            life_time=1.0
        )
    # Print distance-sorted nearby agents so you can verify axes:
    #   ego frame: +x=forward, +y=LEFT.
    #   A car parked to your LEFT should have ey > 0; to your RIGHT ey < 0.
    filled = []
    for i in range(neighbor_np.shape[0]):
        last = neighbor_np[i, -1]
        if np.abs(last).sum() < 1e-6:
            continue
        ex, ey = float(last[0]), float(last[1])
        vx, vy = float(last[4]), float(last[5])
        w,  l  = float(last[6]), float(last[7])
        dist = np.sqrt(ex**2 + ey**2)
        filled.append((dist, i, ex, ey, vx, vy, w, l))
    filled.sort()

    print(f"[agents] {len(filled)}/{neighbor_np.shape[0]} slots filled  (ego +x=fwd, +y=LEFT)")
    for dist, slot, ex, ey, vx, vy, w, l in filled[:6]:
        side  = 'L' if ey >= 0 else 'R'
        speed = np.sqrt(vx**2 + vy**2)
        flag  = " <-- CLOSE" if dist < 5.0 else ""
        print(f"  [{slot:2d}] {dist:5.1f}m  fwd={ex:+6.2f}  {side}={abs(ey):5.2f}  "
              f"spd={speed:.2f}  {w:.1f}x{l:.1f}{flag}")


def diffusion_plan_debug(
    cur: 'TrajectoryPoint',
    destination: 'TrajectoryPoint',
    obs):
    
    # DEBUG: bypass diffusion model, just return A* path
    from utils.map_process import _map_processor
    from utils.coord_utils import standard_to_carla
    
    if _map_processor._global_path is None:
        return []
    
    original_path = _map_processor._global_path  # (M, 2) standard frame

    cur_std_x, cur_std_y, _ = carla_transform_to_standard(cur.x, cur.y, 0.0)
    dists = np.linalg.norm(original_path - np.array([cur_std_x, cur_std_y]), axis=1)
    closest = int(np.argmin(dists))

    ref_dx = original_path[1][0] - original_path[0][0]
    ref_dy = original_path[1][1] - original_path[0][1]
    ref_speed = max(MIN_SPEED, np.sqrt(ref_dx**2 + ref_dy**2) / 0.1)

    path = original_path[closest:]

    
    trajectory = []
    for i in range(len(path)):
        sx, sy = path[i]
        carla_x, carla_y, _ = standard_to_carla(sx, sy, 0.0)
        
        if i == 0:
            speed = cur.speed if cur.speed > MIN_SPEED else ref_speed
            angle = cur.angle
        else:
            prev_sx, prev_sy = path[i-1]
            dx = sx - prev_sx
            dy = sy - prev_sy
            angle_std = np.arctan2(dy, dx)
            _, _, carla_yaw_deg = standard_to_carla(0, 0, np.rad2deg(angle_std))
            angle = np.deg2rad(carla_yaw_deg)
            speed = max(MIN_SPEED, np.sqrt(dx**2 + dy**2) / 0.1)
        
        tp = TrajectoryPoint(Direction.FORWARD, carla_x, carla_y, speed, angle)
        trajectory.append(tp)
    
    
    return trajectory

def debug_goal(world, destination, dest_ego_x, dest_ego_y, x0, y0, h):
    """Draw goal debug points and print sanity info each tick.

    GREEN dot = round-trip (ego→world→CARLA): should land exactly on destination.
    RED dot   = raw destination coords.
    ego_x > 0 = goal is ahead; ego_y < 0 = goal is to the right (CARLA left-handed).
    """
    verify_wx, verify_wy = ego_to_world(dest_ego_x, dest_ego_y, x0, y0, h)
    verify_cx, verify_cy, _ = standard_to_carla(verify_wx, verify_wy, 0.0)
    world.debug.draw_point(
        carla.Location(x=float(verify_cx), y=float(verify_cy), z=0.5),
        size=0.1, color=carla.Color(0, 255, 0), life_time=2.0,
    )
    world.debug.draw_point(
        carla.Location(x=float(destination.x), y=float(destination.y), z=0.5),
        size=0.2, color=carla.Color(255, 0, 0), life_time=2.0,
    )

    # Draw ego +y axis direction: should appear to the LEFT if convention is correct
    left_wx, left_wy = ego_to_world(0.0, 1.0, x0, y0, h)
    left_cx, left_cy, _ = standard_to_carla(left_wx, left_wy, 0.0)
    world.debug.draw_point(
        carla.Location(x=float(left_cx), y=float(left_cy), z=0.5),
        size=0.2, color=carla.Color(0, 0, 255), life_time=2.0,  # BLUE = ego +y direction
    )
    print(f"[goal-debug] BLUE dot is ego +y direction — should be to your LEFT")
    print(
        f"[goal-debug] ego=({dest_ego_x:.2f}, {dest_ego_y:.2f})  "
        f"dist={np.sqrt(dest_ego_x**2 + dest_ego_y**2):.2f}m  "
        f"heading={np.rad2deg(h):.1f}deg  "
        f"round-trip=({verify_cx:.2f},{verify_cy:.2f})  "
        f"raw=({destination.x:.2f},{destination.y:.2f})"
    )


def world_to_ego(wx, wy, x0, y0, h):
    dx, dy = wx - x0, wy - y0
    ex =  np.cos(h) * dx + np.sin(h) * dy
    ey = -np.sin(h) * dx + np.cos(h) * dy
    return ex, ey

# Inverse ego→world: rotate by +h then translate
def ego_to_world(ex, ey, x0, y0, h):
    wx = x0 + np.cos(h) * ex - np.sin(h) * ey
    wy = y0 + np.sin(h) * ex + np.cos(h) * ey
    return wx, wy