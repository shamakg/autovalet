"""
Comprehensive coordinate transform tests for collect_data.py.

Verifies:
  1. inverse_conversion_2d math (forward, right, behind, exact values)
  2. ego_matrix is a valid 4x4 rigid-body transform
  3. target_point is ahead (ego[0] > 0) for parking scenarios
  4. left/right sign convention matches lmdrive_command text
  5. route is ordered near→far and ends near target_point
  6. route[0] is close to the ego origin (car hasn't moved yet)
  7. boxes (parked cars) are at plausible ego-frame positions
  8. speed is non-negative
  9. Saved measurement files from run_001 are internally consistent

Run with:
    cd ~/carla_garage/leaderboard/leaderboard/autovalet
    source ~/envs/simlingo/bin/activate
    python -m pytest testbed_tests/test_collect_data_transforms.py -v -s
"""

import sys, os
sys.path.insert(0, '/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo/Bench2Drive/leaderboard/team_code')
sys.path.insert(0, '/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet')

import gzip, json, pathlib
import pytest
import numpy as np
import ujson
from transfuser_utils import inverse_conversion_2d

from parking_scenarios.parking_scenario_easy import ParkingScenarioEasy
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration
from parking_position import parking_vehicle_locations_Town04

pytestmark = pytest.mark.integration

EPISODE_DIR = pathlib.Path(
    '/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/'
    'vla_adapter/finetune/run_001/data/simlingo/run_001/'
    'routes_training/RouteScenario_parking'
)

# ── helpers ──────────────────────────────────────────────────────────────────

def make_config():
    cfg = ScenarioConfiguration()
    cfg.name = "Parking"; cfg.type = "Parking"
    cfg.town = "Town04_Opt"; cfg.other_actors = None; cfg.route = False
    return cfg


def load_first_measurement(episode_path):
    mfiles = sorted((episode_path / 'measurements').glob('*.json.gz'))
    assert mfiles, f"No measurements in {episode_path}"
    with gzip.open(mfiles[0], 'rt') as f:
        return ujson.load(f), mfiles[0]


def load_all_measurements(episode_path):
    return [
        ujson.load(gzip.open(p, 'rt'))
        for p in sorted((episode_path / 'measurements').glob('*.json.gz'))
    ]


def load_first_boxes(episode_path):
    bfiles = sorted((episode_path / 'boxes').glob('*.json.gz'))
    assert bfiles, f"No boxes in {episode_path}"
    with gzip.open(bfiles[0], 'rt') as f:
        return ujson.load(f)


# ── 1. inverse_conversion_2d math ────────────────────────────────────────────

class TestInverseConversion2D:
    """Unit-test the transform function with known geometry."""

    def test_point_at_ego_origin_maps_to_zero(self):
        ego_xy = np.array([10.0, 20.0])
        yaw = np.deg2rad(45.0)
        result = inverse_conversion_2d(ego_xy, ego_xy, yaw)
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-9)

    def test_forward_point_is_positive_x_ego(self):
        """A point directly ahead (in direction of yaw) maps to ego[0] > 0, ego[1] ≈ 0."""
        ego_xy = np.array([0.0, 0.0])
        yaw = np.deg2rad(90.0)                   # facing +y in world
        ahead_world = np.array([0.0, 5.0])       # +y world = ahead
        result = inverse_conversion_2d(ahead_world, ego_xy, yaw)
        assert result[0] > 4.9, f"Ahead point should give ego[0]>0, got {result}"
        assert abs(result[1]) < 0.1, f"Ahead point should give ego[1]≈0, got {result}"

    def test_right_world_gives_negative_ego_lateral(self):
        """+x world at yaw=90° (right of car) → ego[1] < 0."""
        ego_xy = np.array([0.0, 0.0])
        yaw = np.deg2rad(90.0)
        right_world = np.array([5.0, 0.0])
        result = inverse_conversion_2d(right_world, ego_xy, yaw)
        assert result[1] < -4.9, f"Right point should give ego[1]<0, got {result}"
        assert abs(result[0]) < 0.1, f"Right point should give ego[0]≈0, got {result}"

    def test_left_world_gives_positive_ego_lateral(self):
        """-x world at yaw=90° (left of car) → ego[1] > 0."""
        ego_xy = np.array([0.0, 0.0])
        yaw = np.deg2rad(90.0)
        left_world = np.array([-5.0, 0.0])
        result = inverse_conversion_2d(left_world, ego_xy, yaw)
        assert result[1] > 4.9, f"Left point should give ego[1]>0, got {result}"

    def test_behind_point_is_negative_x_ego(self):
        """A point directly behind → ego[0] < 0."""
        ego_xy = np.array([0.0, 0.0])
        yaw = np.deg2rad(90.0)
        behind_world = np.array([0.0, -5.0])
        result = inverse_conversion_2d(behind_world, ego_xy, yaw)
        assert result[0] < -4.9, f"Behind point should give ego[0]<0, got {result}"

    def test_exact_values_yaw0(self):
        """At yaw=0 (facing +x), the transform is the identity minus translation."""
        ego_xy = np.array([3.0, 4.0])
        yaw = 0.0
        point = np.array([5.0, 4.0])   # 2m ahead in world x
        result = inverse_conversion_2d(point, ego_xy, yaw)
        np.testing.assert_allclose(result, [2.0, 0.0], atol=1e-9)

    def test_is_left_convention_matches_geometry(self):
        """is_left = (target_pt[1] > 0) must match which world-x side the spot is on."""
        ego_xy = np.array([0.0, 0.0])
        yaw = np.deg2rad(90.0)
        # spot to the LEFT (-x world at yaw=90)
        left_spot = np.array([-3.0, 10.0])
        tp_left = inverse_conversion_2d(left_spot, ego_xy, yaw)
        assert tp_left[1] > 0, "Left spot should give target_pt[1] > 0"

        # spot to the RIGHT (+x world at yaw=90)
        right_spot = np.array([3.0, 10.0])
        tp_right = inverse_conversion_2d(right_spot, ego_xy, yaw)
        assert tp_right[1] < 0, "Right spot should give target_pt[1] < 0"


# ── 2. Live CARLA: spawn a vehicle and check measurement fields ───────────────

@pytest.fixture(scope='module')
def live_scenario(carla_world):
    world = carla_world
    scenario = ParkingScenarioEasy(
        world=world, config=make_config(),
        destination=22, parked=[21, 23],
        criteria_enable=False,
    )
    world.tick(); world.tick()
    yield scenario
    try: scenario.cleanup()
    except: pass
    for _ in range(3): world.tick()


class TestLiveTransforms:

    def test_ego_matrix_is_4x4(self, live_scenario):
        actor = live_scenario.car.actor
        m = actor.get_transform().get_matrix()
        assert len(m) == 4 and all(len(row) == 4 for row in m), \
            f"ego_matrix should be 4x4, got shape {len(m)}x{len(m[0])}"

    def test_ego_matrix_rotation_is_orthonormal(self, live_scenario):
        actor = live_scenario.car.actor
        m = np.array(actor.get_transform().get_matrix())
        R = m[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5,
            err_msg="Rotation submatrix of ego_matrix must be orthonormal")

    def test_ego_matrix_last_row(self, live_scenario):
        actor = live_scenario.car.actor
        m = np.array(actor.get_transform().get_matrix())
        np.testing.assert_allclose(m[3], [0, 0, 0, 1], atol=1e-9,
            err_msg="Last row of ego_matrix must be [0,0,0,1]")

    def test_target_point_ahead(self, live_scenario):
        actor = live_scenario.car.actor
        tf = actor.get_transform()
        dest = live_scenario.car.car.destination
        ego_xy = np.array([tf.location.x, tf.location.y])
        yaw = np.deg2rad(tf.rotation.yaw)
        target_pt = inverse_conversion_2d(np.array([dest.x, dest.y]), ego_xy, yaw)
        assert target_pt[0] > 0, \
            f"target_pt[0] must be positive (ahead), got {target_pt[0]:.2f}"

    def test_target_point_reasonable_distance(self, live_scenario):
        actor = live_scenario.car.actor
        tf = actor.get_transform()
        dest = live_scenario.car.car.destination
        ego_xy = np.array([tf.location.x, tf.location.y])
        yaw = np.deg2rad(tf.rotation.yaw)
        target_pt = inverse_conversion_2d(np.array([dest.x, dest.y]), ego_xy, yaw)
        dist = np.linalg.norm(target_pt)
        assert 0.5 < dist < 50.0, \
            f"target_point distance should be 0.5–50m, got {dist:.2f}m"

    def test_is_left_matches_command_filter(self, live_scenario):
        """Verify is_left expression selects only geometrically correct templates."""
        actor = live_scenario.car.actor
        tf = actor.get_transform()
        dest = live_scenario.car.car.destination
        ego_xy = np.array([tf.location.x, tf.location.y])
        yaw = np.deg2rad(tf.rotation.yaw)
        target_pt = inverse_conversion_2d(np.array([dest.x, dest.y]), ego_xy, yaw)

        is_left = target_pt[1] > 0

        dest_loc = parking_vehicle_locations_Town04[22]
        geom_is_left = dest_loc.x < ego_xy[0]

        assert is_left == geom_is_left, \
            f"is_left from target_pt ({is_left}) doesn't match geometry ({geom_is_left})"

    def test_parked_cars_in_ego_frame(self, live_scenario):
        """Parked cars should be at reasonable positions in ego frame."""
        actor = live_scenario.car.actor
        tf = actor.get_transform()
        ego_xy = np.array([tf.location.x, tf.location.y])
        ego_yaw = np.deg2rad(tf.rotation.yaw)

        for parked in live_scenario.parked_cars:
            if not parked.is_alive:
                continue
            pt = parked.get_transform()
            rel = inverse_conversion_2d(
                np.array([pt.location.x, pt.location.y]), ego_xy, ego_yaw
            )
            dist = np.linalg.norm(rel)
            assert dist < 30.0, \
                f"Parked car is {dist:.1f}m away in ego frame — suspiciously far"
            assert dist > 0.5, \
                f"Parked car is {dist:.1f}m away — suspiciously close (inside ego?)"


# ── 3. Saved episode files from run_001 ──────────────────────────────────────

@pytest.fixture(scope='module')
def episode_paths():
    paths = sorted(EPISODE_DIR.glob('Town04_*'))
    assert paths, f"No episodes found under {EPISODE_DIR}"
    return paths


class TestSavedMeasurements:

    def test_all_required_keys_present(self, episode_paths):
        required = {'ego_matrix', 'speed', 'target_point', 'target_point_next',
                    'route', 'route_original', 'command', 'next_command',
                    'lmdrive_command', 'augmentation_translation', 'augmentation_rotation'}
        for ep in episode_paths[:5]:
            m, path = load_first_measurement(ep)
            missing = required - set(m.keys())
            assert not missing, f"{path}: missing keys {missing}"

    def test_command_is_65(self, episode_paths):
        for ep in episode_paths[:5]:
            m, path = load_first_measurement(ep)
            assert m['command'] == 65, f"{path}: command={m['command']}, expected 65"

    def test_speed_non_negative(self, episode_paths):
        for ep in episode_paths[:5]:
            for m in load_all_measurements(ep):
                assert m['speed'] >= 0, f"Negative speed {m['speed']} in {ep}"

    def test_target_point_ahead_in_all_episodes(self, episode_paths):
        failures = []
        for ep in episode_paths:
            m, path = load_first_measurement(ep)
            tp = m['target_point']
            if tp[0] <= 0:
                failures.append(f"{ep.name}: target_pt[0]={tp[0]:.2f}")
        assert not failures, "target_point[0] <= 0 (not ahead):\n" + "\n".join(failures)

    def test_route_length_is_20(self, episode_paths):
        for ep in episode_paths[:5]:
            m, path = load_first_measurement(ep)
            assert len(m['route']) == 20, \
                f"{path}: route has {len(m['route'])} points, expected 20"

    def test_route_ordered_near_to_far(self, episode_paths):
        """Each successive route point should be farther from ego origin."""
        failures = []
        for ep in episode_paths[:5]:
            m, _ = load_first_measurement(ep)
            route = m['route']
            dists = [np.linalg.norm(p) for p in route]
            for i in range(len(dists) - 1):
                if dists[i] > dists[i+1] + 1.0:   # 1m tolerance for curved paths
                    failures.append(
                        f"{ep.name}: route[{i}] dist={dists[i]:.2f} > route[{i+1}] dist={dists[i+1]:.2f}"
                    )
                    break
        assert not failures, "Route not ordered near→far:\n" + "\n".join(failures)

    def test_route_ends_near_target(self, episode_paths):
        """route[-1] should be within 3m of target_point."""
        failures = []
        for ep in episode_paths[:5]:
            m, _ = load_first_measurement(ep)
            route_end = np.array(m['route'][-1])
            target = np.array(m['target_point'])
            dist = np.linalg.norm(route_end - target)
            if dist > 3.0:
                failures.append(f"{ep.name}: route[-1] is {dist:.2f}m from target_point")
        assert not failures, "Route end far from target:\n" + "\n".join(failures)

    def test_route_starts_near_ego_origin(self, episode_paths):
        """route[0] should be within 5m of (0,0) — close to the car."""
        failures = []
        for ep in episode_paths[:5]:
            m, _ = load_first_measurement(ep)
            r0 = np.linalg.norm(m['route'][0])
            if r0 > 5.0:
                failures.append(f"{ep.name}: route[0] dist from ego = {r0:.2f}m")
        assert not failures, "Route start far from ego:\n" + "\n".join(failures)

    def test_lmdrive_command_matches_is_left(self, episode_paths):
        """'left'/'right' in the command text must match target_pt[1] sign."""
        failures = []
        for ep in episode_paths:
            m, path = load_first_measurement(ep)
            tp = m['target_point']
            cmd = m['lmdrive_command'].lower()
            is_left = tp[1] > 0

            if 'left' in cmd and not is_left:
                failures.append(f"{ep.name}: cmd says 'left' but target_pt[1]={tp[1]:.2f} (right)")
            if 'right' in cmd and is_left:
                failures.append(f"{ep.name}: cmd says 'right' but target_pt[1]={tp[1]:.2f} (left)")
        assert not failures, "lmdrive_command direction mismatch:\n" + "\n".join(failures)

    def test_ego_matrix_shape_and_last_row(self, episode_paths):
        for ep in episode_paths[:5]:
            m, path = load_first_measurement(ep)
            mat = np.array(m['ego_matrix'])
            assert mat.shape == (4, 4), f"{path}: ego_matrix shape {mat.shape}"
            np.testing.assert_allclose(mat[3], [0, 0, 0, 1], atol=1e-4,
                err_msg=f"{path}: last row of ego_matrix != [0,0,0,1]")

    def test_ego_matrix_rotation_orthonormal(self, episode_paths):
        for ep in episode_paths[:5]:
            m, path = load_first_measurement(ep)
            R = np.array(m['ego_matrix'])[:3, :3]
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-4,
                err_msg=f"{path}: ego_matrix rotation not orthonormal")


class TestSavedBoxes:

    def test_boxes_have_required_fields(self, episode_paths):
        for ep in episode_paths[:3]:
            boxes = load_first_boxes(ep)
            for box in boxes:
                assert 'class' in box
                assert 'extent' in box
                assert 'position' in box
                assert 'yaw' in box
                assert len(box['extent']) == 3
                assert len(box['position']) == 3

    def test_car_boxes_in_front(self, episode_paths):
        """Parked cars flanking the target spot should be roughly ahead."""
        failures = []
        for ep in episode_paths[:5]:
            boxes = load_first_boxes(ep)
            car_boxes = [b for b in boxes if b['class'] == 'car']
            if not car_boxes:
                failures.append(f"{ep.name}: no car boxes saved")
                continue
            for b in car_boxes:
                pos = b['position']
                dist = np.linalg.norm(pos[:2])
                if dist > 40.0:
                    failures.append(f"{ep.name}: car box {dist:.1f}m away — too far")
                if dist < 0.5:
                    failures.append(f"{ep.name}: car box {dist:.1f}m away — inside ego")
        assert not failures, "Box position issues:\n" + "\n".join(failures)

    def test_box_yaw_is_radian(self, episode_paths):
        """Box yaw should be in radians (|yaw| < 2π), not degrees."""
        for ep in episode_paths[:5]:
            boxes = load_first_boxes(ep)
            for b in boxes:
                assert abs(b['yaw']) < 2 * np.pi + 0.1, \
                    f"{ep.name}: box yaw={b['yaw']:.1f} looks like degrees, not radians"
