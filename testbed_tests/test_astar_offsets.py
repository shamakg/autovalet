"""
Tests that hybrid A* can plan a path for the spawn-offset boundary cases used by
vla_adapter/finetune/collect_data.py.

We test the extreme corners of each episode type's offset range, since those are
the positions most likely to put the ego near a parked car and produce a narrow or
blocked A* search space.

Run with:
    cd ~/carla_garage/leaderboard/leaderboard/autovalet
    source ~/envs/simlingo/bin/activate
    python -m pytest testbed_tests/test_astar_offsets.py -v -s
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration

from parking_scenarios.parking_scenario_easy import ParkingScenarioEasy
from parking_scenarios.parking_scenario_hard import ParkingScenarioHard, HardMode
from testbed.v2_experiment_utils import obstacle_map_from_bbs

pytestmark = pytest.mark.integration

DESTINATION = 22
PARKED      = [21, 23]


def make_config():
    cfg = ScenarioConfiguration()
    cfg.name  = "Parking"
    cfg.type  = "Parking"
    cfg.town  = "Town04_Opt"
    cfg.other_actors = None
    cfg.route = False
    return cfg


def cleanup(scenario, world):
    if scenario:
        try:
            scenario.cleanup()
        except Exception as e:
            print(f"cleanup error: {e}")
    for _ in range(3):
        world.tick()


# Boundary corners for each episode type from collect_data.get_offsets.
# Format: (episode_type_label, y_offset, x_offset)
OFFSET_CASES = [
    # normal:  y=0, x in [-1.5, 1.5]
    ("normal",               0,    -1.5),
    ("normal",               0,     1.5),
    # recovery: y in [10, 55], x in [-2.5, 2.5]
    ("recovery",            10,    -2.5),
    ("recovery",            10,     2.5),
    ("recovery",            55,    -2.5),
    ("recovery",            55,     2.5),
    # pedestrian_normal: y=0, x in [-1.0, 1.0]
    ("pedestrian_normal",    0,    -1.0),
    ("pedestrian_normal",    0,     1.0),
    # pedestrian_recovery: y in [10, 50], x in [-1.0, 1.0]
    ("pedestrian_recovery", 10,    -1.0),
    ("pedestrian_recovery", 50,     1.0),
    # door_normal: y in [48, 55], x in [-1.0, 1.0]
    ("door_normal",         48,    -1.0),
    ("door_normal",         55,     1.0),
]


@pytest.mark.parametrize("label,y_offset,x_offset", OFFSET_CASES,
                         ids=[f"{l}_y{y}_x{x}" for l, y, x in OFFSET_CASES])
def test_astar_finds_path(carla_world, label, y_offset, x_offset):
    """A* must return a non-empty trajectory for every offset boundary case."""
    world    = carla_world
    scenario = None
    try:
        use_ped  = "pedestrian" in label
        use_door = "door"       in label

        if use_ped:
            scenario = ParkingScenarioHard(
                world=world, config=make_config(),
                destination=DESTINATION, parked=PARKED,
                criteria_enable=False,
                mode=HardMode.PedMode,
                start_y_offset=y_offset, start_x_offset=x_offset,
            )
        elif use_door:
            scenario = ParkingScenarioHard(
                world=world, config=make_config(),
                destination=DESTINATION, parked=PARKED,
                criteria_enable=False,
                mode=HardMode.DoorMode,
                start_y_offset=y_offset, start_x_offset=x_offset,
            )
        else:
            scenario = ParkingScenarioEasy(
                world=world, config=make_config(),
                destination=DESTINATION, parked=PARKED,
                criteria_enable=False,
                start_y_offset=y_offset, start_x_offset=x_offset,
            )

        world.tick()
        world.tick()

        # Mirror default_runner's obs setup
        ego_loc = scenario.car.actor.get_location()
        scenario.car.car.obs = obstacle_map_from_bbs(
            scenario.parked_cars_bbs, min_y=ego_loc.y - 5)
        parked_car_ids = {car.id for car in scenario.parked_cars}

        # One planning step
        scenario.car.car.localize()
        scenario.car.run_step(parked_car_ids)

        mode       = scenario.car.car.mode.name
        n_waypoints = len(scenario.car.car.trajectory)

        print(f"  {label:25s} y={y_offset:5.1f}  x={x_offset:5.1f} → "
              f"{n_waypoints} waypoints  mode={mode}")

        assert mode != "FAILED", (
            f"{label} y={y_offset} x={x_offset}: car entered FAILED mode immediately"
        )
        assert n_waypoints > 0, (
            f"{label} y={y_offset} x={x_offset}: A* returned empty trajectory"
        )

    finally:
        cleanup(scenario, world)
