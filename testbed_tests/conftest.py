"""
pytest fixtures for integration tests that require a live CARLA server.

All integration tests share a single world load (session-scoped) so Town04_Opt
is loaded once per test run. Each test is responsible for spawning and cleaning
up its own actors.
"""

import sys
import os
import pytest

# ── path setup ───────────────────────────────────────────────────────────────
AUTOVALET = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARLA_PYTHON = '/home/sumesh/opt/carla/PythonAPI/carla'
SCENARIO_RUNNER = '/home/sumesh/carla_garage/scenario_runner'
LEADERBOARD = '/home/sumesh/carla_garage/leaderboard'

for p in [AUTOVALET, CARLA_PYTHON, SCENARIO_RUNNER, LEADERBOARD]:
    if p not in sys.path:
        sys.path.insert(0, p)

import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from testbed.v2_experiment_utils import load_client, town04_load

DELTA_SECONDS = 0.1


@pytest.fixture(scope='session')
def carla_world():
    """Load Town04_Opt once for the entire test session."""
    client = load_client()
    world = town04_load(client)

    CarlaDataProvider.set_client(client)
    CarlaDataProvider.set_world(world)

    yield world

    # Restore async mode so CARLA is left in a usable state
    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)


def tick(world, n=2):
    for _ in range(n):
        world.tick()
