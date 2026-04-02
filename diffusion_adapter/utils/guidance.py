import os
import torch
import numpy as np
from typing import List, Optional
import sys
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from diffusion_planner.model.guidance.collision import collision_guidance_fn
import math

# ---------------------------------------------------------------------------

# Guidance Function Constants
GOAL_GUIDANCE_SCALE = 0
PATH_GUIDANCE_SCALE = 20000 # tune these
COLLISION_SCALE = 100

# ---------------------------------------------------------------------------

_device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class _CombinedGuidance:
    """
    Guidance function following the GuidanceWrapper pattern:
    1. One-step denoising correction for better x0 estimate
    2. Denormalize state and observations
    3. Apply collision + path-following + goal energy functions
    4. Only active at late diffusion steps (t < 0.1)
    """
    def __init__(self):
        self._goal = None          # (2,) tensor: destination in ego-centric standard frame
        self._path = None          # (M, 2) tensor: A* path in ego-centric standard frame

    def set_goal(self, gx: float, gy: float):
        self._goal = torch.tensor([gx, gy], dtype=torch.float32).to(_device)

    def set_path(self, path_ego: np.ndarray):
        """Set A* path points in ego-centric standard frame. (M, 2)"""
        self._path = torch.tensor(path_ego, dtype=torch.float32).to(_device)

    def __call__(self, x, t, cond, *args, **kwargs):
        
        # ---------------------------------------------------------------------------

        ### Below is borrowed from Guidance Wrapper: diffusion_adapter/Diffusion-Planner/diffusion_planner/model/guidance/guidance_wrapper.py
        inputs = kwargs.get('inputs')
        state_normalizer = kwargs.get('state_normalizer')
        observation_normalizer = kwargs.get('observation_normalizer')
        model = kwargs.get('model')
        model_condition = kwargs.get('model_condition')
        B, P, _ = x.shape

        # --- One-step denoising correction 
        with torch.no_grad():
            x_fix = model(x, t, **model_condition) - x.detach()
        x_fix = x_fix.reshape(B, P, -1, 4)
        x_fix[:, :, 0] = 0.0  # don't correct the current state
        x = x + x_fix.reshape(B, P, -1)

        # --- Denormalize state and observations ---
        x_real = state_normalizer.inverse(x.reshape(B, P, -1, 4))
        inputs_real = observation_normalizer.inverse(inputs)

        # ---------------------------------------------------------------------------

        # use below to remove collision guidance function for testing
        # energy = torch.tensor(0.0, device=x.device, requires_grad=True)

        # Collision guidance (operates on denormalized data) 
        kwargs_no_inputs = {k: v for k, v in kwargs.items() if k != 'inputs'}
        energy = collision_guidance_fn(x_real, t, cond, inputs_real, *args, **kwargs_no_inputs)

        # Path-following guidance: penalize distance from A* at each timestep 
        if self._path is not None and len(self._path) >= 2:
            ego_traj = x_real[:, 0, 1:, :2]  # [B, T, 2] — skip current state
            # Subsample path to limit memory (max 200 points)
            path = self._path
            if len(path) > 200:
                indices = torch.linspace(0, len(path) - 1, 200).long()
                path = path[indices]
            # Distance from each trajectory point to nearest A* path point
            diff = ego_traj[:, :, None, :] - path[None, None, :, :]  # [B, T, M, 2]
            dist_to_path = torch.norm(diff, dim=-1)  # [B, T, M]
            min_dist, _ = dist_to_path.min(dim=-1)   # [B, T]
            path_energy = -PATH_GUIDANCE_SCALE * min_dist.mean()
            energy = energy + path_energy

        # --- Goal guidance: pull final point toward destination ---
        if self._goal is not None:
            ego_final = x_real[:, 0, -1, :2]  # [B, 2]
            dist = torch.norm(ego_final - self._goal[None], dim=-1)  # [B]
            goal_energy = -GOAL_GUIDANCE_SCALE * dist.mean()
            energy = energy + goal_energy   

        return energy


class _CombinedGuidance_v2:
    def __init__(self):
        self._goal = None
        self._path = None
        self._target_speed = 3.0  # m/s target speed
        
    def set_goal(self, gx, gy):
        self._goal = torch.tensor([gx, gy], dtype=torch.float32).to(_device)

    def set_path(self, path_ego):
        self._path = torch.tensor(path_ego, dtype=torch.float32).to(_device)

    def __call__(self, x, t, cond, *args, **kwargs):
        state_normalizer = kwargs.get('state_normalizer')
        B, P, _ = x.shape

        x_real = state_normalizer.inverse(x.reshape(B, P, -1, 4))
        # x_real: [B, P, T+1, 4] where [:, 0, :, :] = ego

        ego_traj = x_real[:, 0, :, :]      # [B, T+1, 4] — ego trajectory
        ego_pos  = ego_traj[:, 1:, :2]     # [B, T, 2]   — skip current state
        neighbor_pos = x_real[:, 1:, 1:, :2]  # [B, P-1, T, 2] — neighbor positions

        energy = torch.tensor(0.0, device=x.device, requires_grad=True)

        # 1. COLLISION AVOIDANCE — paper eq (9)
        # Signed distance between ego and each neighbor at each timestep
        if neighbor_pos.shape[1] > 0:
            r = 3.0       # collision-sensitive distance (meters)
            omega_c = 4.0
            eps = 1e-6

            # [B, P-1, T, 2] - [B, 1, T, 2] = [B, P-1, T, 2]
            diff = neighbor_pos - ego_pos.unsqueeze(1)
            dist = torch.norm(diff, dim=-1)  # [B, P-1, T]

            # Approximate signed distance (positive = no overlap, negative = overlap)
            # Using simple Euclidean distance minus collision radius as proxy
            collision_r = 2.0  # approximate vehicle radius
            signed_dist = dist - collision_r  # positive = safe, negative = collision

            # Paper's energy: penalize when within sensitive distance
            pen = torch.clamp(1.0 - dist / r, min=0.0)  # [B, P-1, T]
            psi = torch.exp(omega_c * pen) - omega_c * pen

            # Normalize by number of active neighbors
            active = (dist < 50.0).float()  # ignore very far agents
            denom = active.sum() + eps
            collision_energy = (psi * active).sum() / denom
            energy = energy + COLLISION_SCALE * collision_energy

        # 2. PATH FOLLOWING
        if self._path is not None and len(self._path) >= 2:
            path = self._path
            if len(path) > 200:
                indices = torch.linspace(0, len(path) - 1, 200).long()
                path = path[indices]
            diff = ego_pos[:, :, None, :] - path[None, None, :, :]  # [B, T, M, 2]
            min_dist, _ = torch.norm(diff, dim=-1).min(dim=-1)       # [B, T]
            path_energy = -PATH_GUIDANCE_SCALE * min_dist.mean()
            energy = energy + path_energy

        # 3. GOAL
        if self._goal is not None:
            ego_final = ego_pos[:, -1, :]  # [B, 2]
            goal_dist = torch.norm(ego_final - self._goal[None], dim=-1)
            goal_energy = -GOAL_GUIDANCE_SCALE * goal_dist.mean()
            energy = energy + goal_energy

        return energy


class _CombinedGuidance_v3:
    def __init__(self):
        self._goal = None
        self._path = None
        
    def set_goal(self, gx, gy):
        self._goal = torch.tensor([gx, gy], dtype=torch.float32).to(_device)

    def set_path(self, path_ego):
        self._path = torch.tensor(path_ego, dtype=torch.float32).to(_device)

    def __call__(self, x, t, cond, *args, **kwargs):
        inputs = kwargs.get('inputs')
        state_normalizer = kwargs.get('state_normalizer')
        observation_normalizer = kwargs.get('observation_normalizer')
        model = kwargs.get('model')
        model_condition = kwargs.get('model_condition')
        B, P, _ = x.shape

        # --- One-step denoising correction ---
        with torch.no_grad():
            x_fix = model(x, t, **model_condition) - x.detach()
        x_fix = x_fix.reshape(B, P, -1, 4)
        x_fix[:, :, 0] = 0.0
        x = x + x_fix.reshape(B, P, -1)

        # --- Denormalize ---
        x_real = state_normalizer.inverse(x.reshape(B, P, -1, 4))
        inputs_real = observation_normalizer.inverse(inputs)

        ego_pos = x_real[:, 0, 1:, :2]  # [B, T, 2] skip current state

        energy = torch.tensor(0.0, device=x.device, requires_grad=True)

        # --- 1. COLLISION AVOIDANCE (against observed agent positions) ---
        agents_past = inputs_real['neighbor_agents_past']  # [B, 32, 21, 11]
        agent_pos = agents_past[:, :, -1, :2]             # [B, 32, 2] most recent

        # Only include agents that are actually present
        active_mask = (agents_past[:, :, -1, :].abs().sum(dim=-1) > 0.1).float()  # [B, 32]

        # Collision energy against each agent at each ego timestep
        diff = ego_pos.unsqueeze(2) - agent_pos.unsqueeze(1)  # [B, T, 32, 2]
        dist = torch.norm(diff, dim=-1)                        # [B, T, 32]

        r = 4.5        # sensitive radius — gradients produced within this distance
        omega_c = 3.0  # sharpness of Ψ function
        eps = 1e-6

        pen = torch.clamp(1.0 - dist / r, min=0.0)            # [B, T, 32]
        psi = torch.exp(omega_c * pen) - omega_c * pen         # Ψ(x) = e^x - x

        # log-sum-exp over agents per timestep: emphasises the closest obstacle
        # like max but with smooth gradients (no abrupt switches between agents).
        active = active_mask.unsqueeze(1).expand_as(dist)      # [B, T, 32]
        large_neg = -1e4
        masked_psi = torch.where(active > 0, psi, torch.full_like(psi, large_neg))
        collision_energy = COLLISION_SCALE * pow(torch.logsumexp(masked_psi, dim=-1).mean(), 2)  # [B, T] → scalar

        energy = energy +  collision_energy

        # --- 2. PATH FOLLOWING ---
        if self._path is not None and len(self._path) >= 2:
            path = self._path
            if len(path) > 200:
                indices = torch.linspace(0, len(path) - 1, 200).long()
                path = path[indices]
            diff = ego_pos[:, :, None, :] - path[None, None, :, :]  # [B, T, M, 2]
            min_dist, _ = torch.norm(diff, dim=-1).min(dim=-1)       # [B, T]
            path_energy = -PATH_GUIDANCE_SCALE * min_dist.mean()
            energy = energy + path_energy

        # --- 3. GOAL ---
        if self._goal is not None:
            ego_final = ego_pos[:, -1, :]
            goal_dist = torch.norm(ego_final - self._goal[None], dim=-1)
            goal_energy = -GOAL_GUIDANCE_SCALE * goal_dist.mean()
            energy = energy + goal_energy

        print(f"[guidance] col={float(collision_energy):.4f} path={float(path_energy) if self._path is not None else 0:.2f} goal={float(goal_energy) if self._goal is not None else 0:.2f}")

        return energy

class _CombinedGuidance_v4:
    def __init__(self):
        self._goal = None          
        self._path = None          

    def set_goal(self, gx: float, gy: float):
        self._goal = torch.tensor([gx, gy], dtype=torch.float32).to(_device)

    def set_path(self, path_ego: np.ndarray):
        self._path = torch.tensor(path_ego, dtype=torch.float32).to(_device)

    def __call__(self, x, t, cond, *args, **kwargs):
        inputs = kwargs.get('inputs')
        state_normalizer = kwargs.get('state_normalizer')
        observation_normalizer = kwargs.get('observation_normalizer')
        model = kwargs.get('model')
        model_condition = kwargs.get('model_condition')
        B, P, _ = x.shape

        # --- One-step denoising correction (x_0 estimate) ---
        with torch.no_grad():
            x_fix = model(x, t, **model_condition) - x.detach()
        x_fix = x_fix.reshape(B, P, -1, 4)
        x_fix[:, :, 0] = 0.0  
        x = x + x_fix.reshape(B, P, -1)

        # --- Denormalize ---
        # x_real: [B, P, T, 4] -> [x, y, sin(h), cos(h)]
        x_real = state_normalizer.inverse(x.reshape(B, P, -1, 4))
        inputs_real = observation_normalizer.inverse(inputs)

        # 1. Collision Guidance
        kwargs_no_inputs = {k: v for k, v in kwargs.items() if k != 'inputs'}
        collision_energy = collision_guidance_fn(x_real, t, cond, inputs_real, *args, **kwargs_no_inputs)
        energy = collision_energy

        # 2. Path-Following Guidance (The Fix)
        if self._path is not None and len(self._path) >= 2:
            ego_pos = x_real[:, 0, 1:, :2]     # [B, T-1, 2]
            ego_heading = x_real[:, 0, 1:, 2:] # [B, T-1, 2] (sin, cos)
            
            path = self._path
            if len(path) > 200:
                indices = torch.linspace(0, len(path)-1, 200).long()
                path = path[indices]

            # Calculate Distance Matrix: [B, T, M]
            diff = ego_pos[:, :, None, :] - path[None, None, :, :]
            dist_sq = torch.sum(diff**2, dim=-1) # Squared distance for smoother gradients
            
            # Use LogSumExp as a "Soft-Min" to avoid sharp jumps between A* points
            # This creates a "smooth tube" instead of individual gravity wells
            path_energy_dist = -PATH_GUIDANCE_SCALE * dist_sq.min(dim=-1)[0].mean()
            
            # --- Directional Alignment (Prevents the "Circles") ---
            # Calculate local tangent of A* path
            path_tangents = path[1:] - path[:-1] # [M-1, 2]
            path_tangents = path_tangents / (torch.norm(path_tangents, dim=-1, keepdim=True) + 1e-6)
            
            # For each traj point, find the direction of the closest path segment
            _, closest_idx = dist_sq[:, :, :-1].min(dim=-1) # [B, T]
            target_tangents = path_tangents[closest_idx]     # [B, T, 2]
            
            # Alignment: dot product of (sin, cos) with (dy, dx)
            # Higher dot product = better alignment
            alignment = torch.sum(ego_heading * target_tangents, dim=-1)
            direction_energy = 10000.0 * alignment.mean() # Tune this scale (5k is a good start)
            
            energy = energy + 0 * path_energy_dist + direction_energy

            print("PATH" ,float(path_energy_dist), "Direction ", float(direction_energy), "Collision ", float(collision_energy))

        # 3. Goal Guidance (Final point attraction)
        if self._goal is not None:
            ego_final = x_real[:, 0, -1, :2]
            dist_sq_goal = torch.sum((ego_final - self._goal[None])**2, dim=-1)
            goal_energy = -GOAL_GUIDANCE_SCALE * dist_sq_goal.mean()
            energy = energy + goal_energy   

        return energy
