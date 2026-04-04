export CARLA_ROOT=/home/sumesh/opt/carla/PythonAPI/carla
export WORK_DIR=/home/sumesh/carla_garage
export CARLA_PORT=2000
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${PYTHONPATH}"

source /opt/ros/humble/setup.bash
export HOST=localhost

/home/sumesh/miniforge3/envs/carla/bin/python3 '/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/runner_test_medium.py'