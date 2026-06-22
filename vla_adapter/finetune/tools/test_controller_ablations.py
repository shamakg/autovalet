"""Controller ablation tests.

Test 1 (--test gt_route):
    Replace model pred_route with the GT route from a training episode.
    If parking succeeds → controller is fine, model prediction is the bottleneck.
    If parking still fails → controller itself is the bottleneck.

Test 2 (--test constant_speed):
    Replace model pred_speed_wps with a constant forward speed profile while
    keeping the model's own route prediction unchanged.
    If car follows route correctly → speed head is the bottleneck.
    If car still doesn't follow → turn controller or route head is the bottleneck.

Usage:
    python test_controller_ablations.py --test gt_route --episode Town04_0000
    python test_controller_ablations.py --test constant_speed --speed 2.0
    python test_controller_ablations.py --test gt_route --episode Town04_0000 --scenario 2
"""

import argparse
import gzip
import json
import os
import pathlib
import signal
import sys

sys.path.insert(0, '/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet')
sys.path.insert(0, '/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter')

import numpy as np
import torch

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from agent_interface import SimLingoAdapter
from benchmark import (
    _DEFAULT_CHECKPOINT, RECORD, run_scenario, make_run_dir,
)
from default_runner import ScenarioMode
from testbed.v2_experiment_utils import load_client, town04_load, town04_spectator_bev
from testbed.recording_utils import make_cleanup_handler
from v2_experiment import SCENARIOS
from team_code.transfuser_utils import inverse_conversion_2d, preprocess_compass

_DEFAULT_DATA_DIR = pathlib.Path(
    '/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter'
    '/finetune/run_001/data/simlingo/parking_ft/routes_training/RouteScenario_parking'
)


# ---------------------------------------------------------------------------
# Ablation adapter subclasses — all new logic lives here
# ---------------------------------------------------------------------------

class GtRouteAdapter(SimLingoAdapter):
    """Closed-loop GT oracle: at each tick, find the nearest GT world frame to the
    live car, take the next N GT world positions, transform them to the live ego
    frame, and feed those as route + speed waypoints.

    This means:
    - Steering is always correct for the live car's actual pose (not the GT car's)
    - Speed target reflects real GT inter-frame distances (not clock-based)
    - Forward/reverse transitions fire based on where the car IS, not elapsed time
    """

    def load_gt_episode(self, episode_dir: pathlib.Path):
        meas_dir = episode_dir / 'measurements'
        if not meas_dir.exists():
            raise FileNotFoundError(f"No measurements dir in {episode_dir}")

        meas_files = sorted(meas_dir.glob('*.json.gz'))
        frames = []
        for mf in meas_files:
            with gzip.open(mf, 'rt') as f:
                frames.append(json.load(f))

        # GT world positions in CARLA coords (ego_matrix[:2, 3])
        self._gt_world_xy = np.array([
            np.array(f['ego_matrix'])[:2, 3] for f in frames
        ], dtype=np.float32)  # (T, 2)

        self._gt_n_frames = len(frames)
        self._gt_idx = 0  # nearest GT frame index — only moves forward

        # N from route field (excluding the [0,0] origin entry)
        N = len(frames[0]['route']) - 1
        self._gt_N = N

        print(f"[gt_route] Loaded {len(frames)} GT frames from {episode_dir.name}, N={N}, "
              f"start=({self._gt_world_xy[0,0]:.1f},{self._gt_world_xy[0,1]:.1f}), "
              f"end=({self._gt_world_xy[-1,0]:.1f},{self._gt_world_xy[-1,1]:.1f})")

    def control_pid(self, route_waypoints, velocity, speed_waypoints):
        if not hasattr(self, '_gt_world_xy'):
            return super().control_pid(route_waypoints, velocity, speed_waypoints)

        # Live car pose in CARLA world frame
        actor = self.hero_actor
        loc = actor.get_location()
        yaw = np.deg2rad(actor.get_transform().rotation.yaw)
        compass = preprocess_compass(yaw + np.deg2rad(90.0))
        cur_xy = np.array([loc.x, loc.y], dtype=np.float32)

        # Advance GT index to the nearest frame (monotonically — never go backward)
        traj = self._gt_world_xy
        idx = self._gt_idx
        while idx + 1 < self._gt_n_frames:
            dc = np.sum((traj[idx] - cur_xy) ** 2)
            dn = np.sum((traj[idx + 1] - cur_xy) ** 2)
            if dn < dc:
                idx += 1
            else:
                break
        self._gt_idx = idx

        # Next N GT world positions as future targets
        N = self._gt_N
        future_wxy = np.array([
            traj[min(idx + j + 1, self._gt_n_frames - 1)] for j in range(N)
        ], dtype=np.float32)  # (N, 2)

        # Transform from world to live ego frame
        ego_pts = np.stack([
            inverse_conversion_2d(future_wxy[j], cur_xy, compass)
            for j in range(N)
        ]).astype(np.float32)  # (N, 2)

        gt_route = torch.from_numpy(ego_pts[np.newaxis])  # (1, N, 2)
        gt_sw    = torch.from_numpy(ego_pts[np.newaxis])  # speed = same future positions

        # Expose for recording overlay
        self.latest_pred_route = ego_pts

        progress_pct = 100 * idx / max(self._gt_n_frames - 1, 1)
        print(f"[gt_route] idx={idx}/{self._gt_n_frames-1} ({progress_pct:.0f}%) "
              f"world=({cur_xy[0]:.1f},{cur_xy[1]:.1f}) "
              f"ego[0]=({ego_pts[0,0]:.3f},{ego_pts[0,1]:.3f}) "
              f"ego[-1]=({ego_pts[-1,0]:.3f},{ego_pts[-1,1]:.3f})")

        return super().control_pid(gt_route, velocity, gt_sw)


class AStarAdapter(SimLingoAdapter):
    """Feeds the v2 A* planner trajectory into the agent_interface PID controller.

    The benchmark loop (enable_astar=True) runs the v2 Car's perceive()/plan() with
    its real replan gating, then hands the resulting trajectory here. The controller
    under test is agent_interface's run_step_testbed_astar -> control_pid (same PID
    the VLA model uses), so this isolates the controller given a known-good path.

    Key fix vs. the swerving version: the controller's waypoint index (_astar_ti) is
    reset ONLY when plan() actually produced a new trajectory (identity change), not
    every tick. Re-seeking from index 0 each frame was causing the swerving.
    """

    _last_astar_traj_obj = None

    def set_astar_trajectory(self, traj_points):
        # plan() reassigns Car.trajectory only on a real replan, so identity is stable
        # between replans. Reset the controller's progress index only when it changes.
        if traj_points is self._last_astar_traj_obj:
            return
        self._last_astar_traj_obj = traj_points
        if traj_points:
            self._astar_trajectory = list(traj_points)
            self._astar_ti = 0
            self.latest_astar_trajectory = np.array(
                [[p.x, p.y] for p in traj_points], dtype=np.float32
            )
        else:
            self._astar_trajectory = []
            self._astar_ti = 0
            self.latest_astar_trajectory = None


class ConstantSpeedAdapter(SimLingoAdapter):
    """Overrides model speed_waypoints with a constant forward speed profile."""

    def set_target_speed(self, speed_mps: float):
        self._target_speed_mps = speed_mps
        print(f"[constant_speed] Target speed: {speed_mps:.2f} m/s")

    def control_pid(self, route_waypoints, velocity, speed_waypoints):
        if hasattr(self, '_target_speed_mps'):
            dt = self.config.data_save_freq / self.config.carla_fps
            N = speed_waypoints.shape[1]
            fake_sw = torch.zeros_like(speed_waypoints)
            fake_sw[0, :, 0] = (torch.arange(1, N + 1, dtype=torch.float32,
                                              device=speed_waypoints.device)
                                 * (self._target_speed_mps * dt))
            return super().control_pid(route_waypoints, velocity, fake_sw)
        return super().control_pid(route_waypoints, velocity, speed_waypoints)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Controller ablation tests")
    parser.add_argument('--test', required=True, choices=['gt_route', 'constant_speed', 'astar'])
    parser.add_argument('--episode', type=str, default=None,
                        help="[gt_route] Episode dir name, e.g. Town04_0000")
    parser.add_argument('--speed', type=float, default=2.0,
                        help="[constant_speed] Target speed in m/s (default: 2.0)")
    parser.add_argument('--scenario', type=int, default=1,
                        help="1-indexed scenario from SCENARIOS (default: 1)")
    parser.add_argument('--mode', type=str, default='empty',
                        choices=[m.value for m in ScenarioMode],
                        help="default 'empty' = no parked cars/cones (clean controller test)")
    parser.add_argument('--checkpoint', type=str, default=_DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    scenario_idx = args.scenario - 1
    if not (0 <= scenario_idx < len(SCENARIOS)):
        sys.exit(f"--scenario must be 1–{len(SCENARIOS)}")
    destination_parking_spot, parked_spots = SCENARIOS[scenario_idx]
    scenario_mode = ScenarioMode(args.mode)

    enable_astar = False

    if args.test == 'gt_route':
        if args.episode is None:
            sys.exit("--episode is required for gt_route test")
        episode_dir = _DEFAULT_DATA_DIR / args.episode
        if not episode_dir.exists():
            sys.exit(f"Episode not found: {episode_dir}")
        adapter_class = GtRouteAdapter
        adapter_init_fn = lambda a: a.load_gt_episode(episode_dir)
        print(f"[gt_route] Episode: {args.episode}")

    elif args.test == 'astar':
        adapter_class = AStarAdapter
        adapter_init_fn = None
        enable_astar = True
        print("[astar] A* planner + PID controller (no model inference)")

    else:  # constant_speed
        adapter_class = ConstantSpeedAdapter
        adapter_init_fn = lambda a: a.set_target_speed(args.speed)
        print(f"[constant_speed] {args.speed:.2f} m/s")

    run_dir = make_run_dir()
    all_recording_paths = []
    car_list = [None]
    handler = make_cleanup_handler(lambda: car_list[0], all_recording_paths)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    print(f"Results → {run_dir}")

    client = load_client()
    world = town04_load(client)
    CarlaDataProvider.set_client(client)
    CarlaDataProvider.set_world(world)
    town04_spectator_bev(world)

    ious, weighted_ious = [], []
    collisions_ref, near_misses_ref, walker_collisions_ref = [0], [0], [0]
    recording_path = None
    if RECORD:
        recording_path = os.path.join(run_dir, 'scenario_1.mp4')
        all_recording_paths += [recording_path.replace('.mp4', s)
                                 for s in ('_chase.mp4', '_topdown.mp4')]

    print(f"Scenario {args.scenario}: destination={destination_parking_spot}")
    run_scenario(
        world, destination_parking_spot, parked_spots,
        ious, weighted_ious, collisions_ref, near_misses_ref,
        actual_collisions=[], walker_collisions_ref=walker_collisions_ref,
        recording_path=recording_path, car_list=car_list,
        scenario_mode=scenario_mode, checkpoint_path=args.checkpoint,
        adapter_class=adapter_class, adapter_init_fn=adapter_init_fn,
        enable_astar=enable_astar,
    )

    print(f"\n--- Results ({args.test}) ---")
    if ious:
        print(f"IOU: {ious[0]:.4f}  Collisions: {collisions_ref[0]}")


if __name__ == '__main__':
    main()
