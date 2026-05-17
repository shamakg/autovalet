export CARLA_ROOT=/home/sumesh/opt/carla/PythonAPI/carla
export WORK_DIR=/home/sumesh/carla_garage
export CARLA_PORT=2000
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${PYTHONPATH}"

source /opt/ros/humble/setup.bash
export HOST=localhost

source /home/sumesh/envs/simlingo/bin/activate
python /home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/diffusion_adapter/benchmark.py