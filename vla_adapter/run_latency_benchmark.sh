#!/bin/bash

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Activate environment
source /home/sumesh/envs/simlingo/bin/activate

# Paths
export CARLA_ROOT=/home/sumesh/opt/carla
export SCENARIO_RUNNER_ROOT=/home/sumesh/carla_garage/scenario_runner
export LEADERBOARD_ROOT=/home/sumesh/carla_garage/leaderboard
export SIMLINGO_ROOT=$SCRIPT_DIR/simlingo

# HuggingFace cache — point to local pretrained dir so models aren't re-downloaded
export HUGGINGFACE_HUB_CACHE='/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/model/pretrained'
export TRANSFORMERS_CACHE='/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/model/pretrained'
export HF_HOME='/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/model/pretrained'

# PYTHONPATH
# Add carla agents, team_code, and simlingo base
export PYTHONPATH="/home/sumesh/opt/carla/PythonAPI/carla:$SCENARIO_RUNNER_ROOT:$LEADERBOARD_ROOT:$SIMLINGO_ROOT:$SIMLINGO_ROOT/team_code:$SCRIPT_DIR:$PYTHONPATH"

# Run baseline (0ms)
# python3 "$SCRIPT_DIR/benchmark_latency.py" --no-comparison --latency 0

echo "Starting VLA Latency Evaluation Suite..."
echo "Total Scenarios: 124"

# Run dynamic latency (directly from SimLingo model)
# python3 "$SCRIPT_DIR/benchmark_latency.py" --no-comparison --latency 1 --dynamic-latency --mode STOP_EARLY
# python3 "$SCRIPT_DIR/benchmark_latency.py" --no-comparison --latency 1 --dynamic-latency --mode MISS
# python3 "$SCRIPT_DIR/benchmark_latency.py" --no-comparison --latency 1 --dynamic-latency --mode COLLIDE
python3 "$SCRIPT_DIR/benchmark_latency.py" --no-comparison --dynamic-latency ## DoorMode




echo ""
echo "========================================"
echo " LATENCY COMPARISON REPORT"
echo "========================================"
echo "Results saved to: $SCRIPT_DIR/results/"
echo "Check the subdirectories for lat0ms and dynamic latencies."
