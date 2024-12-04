export CARLA_ROOT=/workspace/PythonAPI/carla
export WORK_DIR=/workspace
export CARLA_PORT=2000 
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${PYTHONPATH}"

source /opt/ros/foxy/setup.bash
export HOST=host.docker.internal

python3 '/workspace/leaderboard/leaderboard/autovalet/runner_test_medium.py'