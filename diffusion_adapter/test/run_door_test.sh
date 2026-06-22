#!/bin/bash
export CARLA_ROOT=/home/sumesh/opt/carla/PythonAPI/carla
export WORK_DIR=/home/sumesh/carla_garage
export CARLA_PORT=2000
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export SIMLINGO_TRAINING=/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo
export PYTHONPATH="${CARLA_ROOT}:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${SIMLINGO_TRAINING}:${PYTHONPATH}"

source /home/sumesh/envs/simlingo/bin/activate

python3 /home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/diffusion_adapter/test/test_door_safety_metric.py
