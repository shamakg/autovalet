#!/bin/bash
# Launches CARLA server, runs the VLA benchmark, saves all recordings, then kills CARLA.

set -e

CARLA_SERVER=/home/sumesh/opt/carla/CarlaUE4.sh
CARLA_PORT=2000

# --- Start CARLA in the background ---
echo "Starting CARLA server..."
$CARLA_SERVER -RenderOffScreen -carla-port=$CARLA_PORT &
CARLA_PID=$!
echo "CARLA PID: $CARLA_PID"

# Give CARLA time to boot
sleep 15

# --- Run the benchmark ---
echo "Running benchmark..."

# Source env and run benchmark; pipe "n" to auto-skip the delete prompt (keep all recordings)
(
  export CARLA_ROOT=/home/sumesh/opt/carla/PythonAPI/carla
  export WORK_DIR=/home/sumesh/carla_garage
  export CARLA_PORT=$CARLA_PORT
  export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
  export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
  export SIMLINGO_ROOT=/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo
  export SIMLINGO_TEAM_CODE=/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo/team_code
  export SIMLINGO_TRAINING=/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo
  export PYTHONPATH="${CARLA_ROOT}:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${SIMLINGO_TRAINING}:${SIMLINGO_TEAM_CODE}:${PYTHONPATH}"
  source /opt/ros/humble/setup.bash
  export HOST=localhost
  export HUGGINGFACE_HUB_CACHE='/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/model/pretrained'
  export TRANSFORMERS_CACHE='/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/model/pretrained'
  export HF_HOME='/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/model/pretrained'
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1

  source /home/sumesh/envs/simlingo/bin/activate

  cd ~/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/model
  echo "n" | python3 /home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/benchmark.py
)

BENCH_EXIT=$?
echo "Benchmark exited with code: $BENCH_EXIT"

# --- Kill CARLA ---
echo "Stopping CARLA (PID $CARLA_PID)..."
kill -SIGINT $CARLA_PID 2>/dev/null || true
sleep 3
# Force kill if still alive
kill -9 $CARLA_PID 2>/dev/null || true

echo "Done."
