export CARLA_ROOT=/home/shamakg/opt/carla/PythonAPI/carla
export WORK_DIR=/home/shamakg/carla_garage
export CARLA_PORT=2000
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export SIMLINGO_ROOT=/home/shamakg/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo
export SIMLINGO_TEAM_CODE=/home/shamakg/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo/team_code
export SIMLINGO_TRAINING=/home/shamakg/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo
export PYTHONPATH="${CARLA_ROOT}:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${SIMLINGO_TRAINING}:${SIMLINGO_TEAM_CODE}:${PYTHONPATH}"
source /opt/ros/humble/setup.bash
export HOST=localhost
export HUGGINGFACE_HUB_CACHE='/home/shamakg/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/model/pretrained'
export TRANSFORMERS_CACHE='/home/shamakg/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/model/pretrained'
export HF_HOME='/home/shamakg/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/model/pretrained'
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source /home/shamakg/envs/simlingo/bin/activate

cd /home/shamakg/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/model

python3 /home/shamakg/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/finetune/test_controller_ablations.py "$@"
