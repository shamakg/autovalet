"""
Agent interface that takes the VLA model's predicted trajectory
and follows it using v2.py's PID controller instead of the model's own PID.
"""

import sys
import os
import importlib.util

import carla
import numpy as np
from agent_interface import SimLingoAdapter


# Explicitly load vla_adapter/v2.py to avoid getting autovalet/v2.py from sys.modules
_v2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v2.py')
_v2_spec = importlib.util.spec_from_file_location('v2_vla', _v2_path)
_v2 = importlib.util.module_from_spec(_v2_spec)
_v2_spec.loader.exec_module(_v2)

TrajectoryPoint = _v2.TrajectoryPoint
Direction = _v2.Direction
mps_to_kmph = _v2.mps_to_kmph
MAX_SPEED = _v2.MAX_SPEED
LOOKAHEAD = _v2.LOOKAHEAD

from v2_controller import VehiclePIDController

STOP_CONTROL = carla.VehicleControl(brake=1.0)
DESTINATION_THRESHOLD = 0.3


class PIDAdapter(SimLingoAdapter):
    def init_testbed(self, checkpoint_path, world, actor, destination, angle, save_path='/tmp/simlingo'):
        super().init_testbed(checkpoint_path, world, actor, destination, angle, save_path)
        self.controller = VehiclePIDController(
            {'K_P': 5, 'K_I': 0.05, 'K_D': 0.2, 'dt': 0.05},
            {'K_P': 5, 'K_I': 0.05, 'K_D': 0.0, 'dt': 0.05},
        )
        self.trajectory = []
        self.ti = 0

    def run_step_testbed(self, timestamp):
        """Run the VLA model to get a trajectory, then follow it with v2's PID."""
        # Run the parent to get the model prediction (this also updates self.latest_pred_route)
        model_control = super().run_step_testbed(timestamp)

        actor = self._cam.parent
        loc = actor.get_location()
        vel = actor.get_velocity()
        transform = actor.get_transform()
        yaw = np.deg2rad(transform.rotation.yaw)
        speed = np.sqrt(vel.x**2 + vel.y**2)

        # Build current position as a TrajectoryPoint
        cur = TrajectoryPoint(Direction.FORWARD, loc.x, loc.y, speed, yaw)

        # If the model produced a trajectory, convert ego-frame waypoints to world TrajectoryPoints
        if self.latest_pred_route is not None and len(self.latest_pred_route) > 0:
            pred = self.latest_pred_route  # Nx2 array in ego frame (forward, left)
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)

            self.trajectory = []
            for i in range(len(pred)):
                wx = loc.x + pred[i, 0] * cos_y - pred[i, 1] * sin_y
                wy = loc.y + pred[i, 0] * sin_y + pred[i, 1] * cos_y
                wp_angle = np.arctan2(wy - loc.y, wx - loc.x) if i == 0 else \
                           np.arctan2(wy - self.trajectory[-1].y, wx - self.trajectory[-1].x)
                self.trajectory.append(
                    TrajectoryPoint(Direction.FORWARD, wx, wy, mps_to_kmph(MAX_SPEED) / 3.6, wp_angle)
                )
            self.ti = 0

        # If no trajectory yet, brake
        if not self.trajectory:
            return STOP_CONTROL

        # --- Follow trajectory with v2's PID (same logic as Car.control) ---
        trajectory = self.trajectory
        ti = self.ti
        wp = trajectory[ti]
        wp_dist = np.sqrt((cur.x - wp.x)**2 + (cur.y - wp.y)**2)

        # Advance to closest waypoint
        for i in range(ti + 1, len(trajectory)):
            d = np.sqrt((cur.x - trajectory[i].x)**2 + (cur.y - trajectory[i].y)**2)
            if d > wp_dist:
                break
            ti = i
            wp = trajectory[i]
            wp_dist = d
        self.ti = ti

        # Find lookahead waypoint
        future_wp = wp
        for i in range(ti + 1, min(ti + LOOKAHEAD + 1, len(trajectory))):
            d = np.sqrt((cur.x - trajectory[i].x)**2 + (cur.y - trajectory[i].y)**2)
            if d < wp_dist:
                break
            future_wp = trajectory[i]
            wp_dist = d

        cur.direction = wp.direction
        ctrl = self.controller.run_step(
            mps_to_kmph(cur.speed),
            mps_to_kmph(wp.speed),
            cur,
            future_wp,
            wp.direction == Direction.REVERSE,
        )
        return ctrl
