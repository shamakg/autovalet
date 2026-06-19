#!/bin/bash
export CARLA_ROOT=/home/shamakg/opt/carla/PythonAPI/carla
export WORK_DIR=/home/shamakg/carla_garage
export CARLA_PORT=2000
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export SIMLINGO_TRAINING=/home/shamakg/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo
export PYTHONPATH="${CARLA_ROOT}:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${SIMLINGO_TRAINING}:${PYTHONPATH}"

source /home/shamakg/envs/simlingo/bin/activate

python3 /home/shamakg/carla_garage/leaderboard/leaderboard/autovalet/diffusion_adapter/test/test_door_safety_metric.py
