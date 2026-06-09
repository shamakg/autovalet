"""
Verify coordinate axis assumptions in collect_data.save_measurement.

Checks:
  1. target_pt[0] > 0  — destination is AHEAD of spawn
  2. Left/right sign — a point placed 5m to the car's right gives ego[1] > 0
     and 5m to the left gives ego[1] < 0 (confirms is_left = target_pt[1] < 0)
  3. is_left agrees with the actual world geometry of spot 22

Run with:
    cd ~/carla_garage/leaderboard/leaderboard/autovalet
    source ~/envs/simlingo/bin/activate
    python -m pytest testbed_tests/test_collect_data_axes.py -v -s
"""

import sys
sys.path.insert(0, '/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo/Bench2Drive/leaderboard/team_code')

import pytest
import numpy as np
from transfuser_utils import inverse_conversion_2d
from parking_scenarios.parking_scenario_easy import ParkingScenarioEasy
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration
from parking_position import parking_vehicle_locations_Town04

pytestmark = pytest.mark.integration


def make_config():
    cfg = ScenarioConfiguration()
    cfg.name = "Parking"; cfg.type = "Parking"
    cfg.town = "Town04_Opt"; cfg.other_actors = None; cfg.route = False
    return cfg


@pytest.fixture(scope="module")
def axes_data(carla_world):
    world = carla_world
    scenario = ParkingScenarioEasy(
        world=world, config=make_config(),
        destination=22, parked=[21, 23],
        criteria_enable=False,
    )
    world.tick(); world.tick()

    actor     = scenario.car.actor
    transform = actor.get_transform()
    dest      = scenario.car.car.destination

    ego_xy = np.array([transform.location.x, transform.location.y])
    yaw    = np.deg2rad(transform.rotation.yaw)

    print(f"\nEgo world   : x={ego_xy[0]:.2f}  y={ego_xy[1]:.2f}  yaw={transform.rotation.yaw:.1f}°")
    print(f"Dest world  : x={dest.x:.2f}  y={dest.y:.2f}")

    target_pt   = inverse_conversion_2d(np.array([dest.x, dest.y]), ego_xy, yaw)
    # known right = +x world (car faces +y at yaw≈90)
    right_ego   = inverse_conversion_2d(ego_xy + np.array([5.0, 0.0]), ego_xy, yaw)
    left_ego    = inverse_conversion_2d(ego_xy + np.array([-5.0, 0.0]), ego_xy, yaw)

    print(f"target_pt ego: [{target_pt[0]:.2f}, {target_pt[1]:.2f}]")
    print(f"right (+x world) → ego[1] = {right_ego[1]:.2f}")
    print(f"left  (-x world) → ego[1] = {left_ego[1]:.2f}")

    yield dict(target_pt=target_pt, right_ego=right_ego, left_ego=left_ego,
               ego_xy=ego_xy, dest=dest)

    try: scenario.cleanup()
    except: pass
    for _ in range(3): world.tick()


def test_destination_is_ahead(axes_data):
    """target_pt[0] must be positive — destination is in front of the car."""
    tp = axes_data["target_pt"]
    assert tp[0] > 0, f"Expected target_pt[0] > 0, got {tp[0]:.2f}"


def test_right_world_gives_negative_ego_lateral(axes_data):
    """+x world (right of car at yaw≈90) gives ego[1] < 0 (inverse_conversion_2d convention)."""
    assert axes_data["right_ego"][1] < 0, \
        f"Expected right → ego[1] < 0, got {axes_data['right_ego'][1]:.2f}"


def test_left_world_gives_positive_ego_lateral(axes_data):
    """-x world (left of car at yaw≈90) gives ego[1] > 0 (inverse_conversion_2d convention)."""
    assert axes_data["left_ego"][1] > 0, \
        f"Expected left → ego[1] > 0, got {axes_data['left_ego'][1]:.2f}"


def test_is_left_expression(axes_data):
    """Confirm is_left = (target_pt[1] > 0) matches actual geometry of spot 22."""
    ego_xy    = axes_data["ego_xy"]
    target_pt = axes_data["target_pt"]
    dest_loc  = parking_vehicle_locations_Town04[22]

    geom_is_left  = dest_loc.x < ego_xy[0]   # spot at lower x → left of car
    code_is_left  = target_pt[1] > 0          # collect_data.py: left → positive ego[1]

    print(f"\nSpot 22 world x={dest_loc.x:.2f}  ego x={ego_xy[0]:.2f}")
    print(f"Geometry says is_left={geom_is_left}, code gives is_left={code_is_left}")

    assert code_is_left == geom_is_left, \
        f"is_left mismatch: geometry={geom_is_left}, code={code_is_left}"
