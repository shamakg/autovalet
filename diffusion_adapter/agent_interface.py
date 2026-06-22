"""
diffusion_agent_interface.py — Minimal adapter for the benchmark.

Replaces SimLingoAdapter. No VLA, no A*, no camera, no Kalman filter.

Usage in benchmark:
    adapter = DiffusionAdapter()
    adapter.init_testbed(ckpt_path, world, actor, destination, dest_angle)
    # each tick:
    control = adapter.run_step_testbed(timestamp)
    done    = adapter.is_done(destination)
"""

import sys
sys.path.insert(0, '/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet')
sys.path.insert(0, '/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/diffusion_adapter')
sys.path.insert(0, '/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/diffusion_adapter/nuplan-devkit')

import carla
import numpy as np

from v2 import TrajectoryPoint, Direction, VehiclePIDController
import diff_adapter as da
from utils.agent_process import AgentHistoryBuffer, AgentState
from utils.coord_utils import carla_transform_to_standard, carla_velocity_to_standard

DIFFUSION_CKPT = '/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/diffusion_adapter/checkpoints/model.pth'
DIFFUSION_ARGS = '/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/diffusion_adapter/checkpoints/args.json'

TARGET_SPEED       = 1.5   # m/s
MAX_LOOKAHEAD_DIST = 12.0  # pure pursuit: use farthest trajectory waypoint within this range

# DiffusionPlanner was trained on nuPlan data where most agents are moving.
# Parked cars (speed ≈ 0) in adjacent parking spots cause the model to suppress
# its trajectory — it interprets nearby stationary agents as blocking obstacles.
# Only pass agents that are actually moving to avoid this.
_MIN_AGENT_SPEED_MS = 0.5   # m/s — below this, exclude from model inputs


def _carla_actors_to_agent_states(world, ego_id):
    """
    Return AgentState objects for moving vehicles and pedestrians in the scene,
    excluding the ego and stationary parked cars (speed < _MIN_AGENT_SPEED_MS).
    """
    states = []
    for actor in world.get_actors():
        type_id = actor.type_id
        if 'vehicle' in type_id:
            if actor.id == ego_id:
                continue
            agent_type = 'vehicle'
        elif 'walker.pedestrian' in type_id:
            agent_type = 'pedestrian'
        else:
            continue

        vel = actor.get_velocity()
        speed = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5
        if speed < _MIN_AGENT_SPEED_MS:
            continue   # skip parked/stationary actors

        loc = actor.get_location()
        yaw = actor.get_transform().rotation.yaw   # CARLA degrees
        bb  = actor.bounding_box

        # passthrough: carla_transform_to_standard is identity except deg→rad
        std_x, std_y, std_h = carla_transform_to_standard(loc.x, loc.y, yaw)
        std_vx, std_vy       = carla_velocity_to_standard(vel.x, vel.y)

        states.append(AgentState(
            actor_id   = actor.id,
            x          = std_x,
            y          = std_y,
            heading    = std_h,
            vx         = std_vx,
            vy         = std_vy,
            width      = bb.extent.y * 2,   # CARLA extent is half-size
            length     = bb.extent.x * 2,
            agent_type = agent_type,
        ))
    return states


class DiffusionAdapter:

    def init_testbed(self, checkpoint_path, world, actor, destination, dest_angle):
        self.actor        = actor
        self.world        = world
        self._destination = destination
        self._dest_angle  = dest_angle
        self.latest_pred_route = None
        self._agent_buf   = AgentHistoryBuffer()

        loc   = actor.get_location()
        yaw   = np.deg2rad(actor.get_transform().rotation.yaw)
        vel   = actor.get_velocity()
        fwd   = actor.get_transform().get_forward_vector()
        speed = vel.x * fwd.x + vel.y * fwd.y + vel.z * fwd.z
        cur = TrajectoryPoint(Direction.FORWARD, loc.x, loc.y, max(0.0, speed), yaw)

        # PID controller from v2.py — same one Car uses
        self.controller = VehiclePIDController(
            {'K_P': 8, 'K_I': 0.05, 'K_D': 0.2, 'dt': 0.05},
            {'K_P': 3, 'K_I': 0.05, 'K_D': 0.0, 'dt': 0.05},
        )

        da.load_model(DIFFUSION_CKPT, DIFFUSION_ARGS)
        da.init_scenario(cur, destination, None)
        print(f"[DiffusionAdapter] Ready. dest=({destination.x:.1f},{destination.y:.1f})")

    def run_step_testbed(self, timestamp):
        actor = self.actor
        loc   = actor.get_location()
        yaw   = np.deg2rad(actor.get_transform().rotation.yaw)
        vel   = actor.get_velocity()
        fwd   = actor.get_transform().get_forward_vector()
        speed = vel.x * fwd.x + vel.y * fwd.y + vel.z * fwd.z

        cur = TrajectoryPoint(Direction.FORWARD, loc.x, loc.y, max(0.0, speed), yaw)

        agent_states = _carla_actors_to_agent_states(self.world, self.actor.id)
        self._agent_buf.update(agent_states)

        traj = da.diffusion_plan(cur, self._destination, None, self._agent_buf)
        if not traj:
            return carla.VehicleControl(brake=1.0)

        # Convert trajectory to ego coordinates for the overlay visualization
        ego_route = []
        for p in traj[:20]:
            dx = p.x - loc.x
            dy = p.y - loc.y
            cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
            ex = dx * cos_y - dy * sin_y
            ey = dx * sin_y + dy * cos_y
            ego_route.append([ex, ey])
        self.latest_pred_route = np.array(ego_route)

        # Pure pursuit on the model's predicted trajectory.
        # Walk forward until the first waypoint >= MAX_LOOKAHEAD_DIST away;
        # fall back to the last waypoint if all are closer.
        lookahead = traj[-1]
        for pt in traj:
            if np.hypot(pt.x - loc.x, pt.y - loc.y) >= MAX_LOOKAHEAD_DIST:
                lookahead = pt
                break

        ctrl = self.controller.run_step(
            speed * 3.6,
            TARGET_SPEED * 3.6,
            cur,
            lookahead,
            False,
        )
        print(f"[diffusion] v={speed:.2f} steer={ctrl.steer:.3f} thr={ctrl.throttle:.3f} "
              f"wp=({lookahead.x:.1f},{lookahead.y:.1f})")
        return ctrl

    def is_done(self, destination):
        loc      = self.actor.get_location()
        yaw      = np.deg2rad(self.actor.get_transform().rotation.yaw)
        half_len = self.actor.bounding_box.extent.x
        front_x  = loc.x + half_len * np.cos(yaw)
        front_y  = loc.y + half_len * np.sin(yaw)
        dx       = front_x - self._destination.x
        dy       = front_y - self._destination.y
        along    = dx * np.cos(self._dest_angle) + dy * np.sin(self._dest_angle)
        print(f"DEPTH TO FAR EDGE {along:.2f}m")
        return along > -0.5

    def destroy_cam(self):
        pass   # no camera