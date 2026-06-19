#!/bin/bash
# Bench2Drive evaluation without SLURM.
# Requires CARLA server already running on $PORT.
#
# Usage:
#   bash bench2drive_eval.sh [devtest|full]   (default: devtest)
#
#   devtest — 3 routes in Town12 (ParkingExit + junctions). NOTE: Town12 may be
#             out-of-distribution for the base model. Bad for overfitting checks.
#   sample  — 5 routes from bench2drive_split (actual simlingo eval towns)
#   full    — all bench2drive_split routes, runs them sequentially

set -e

MODE=${1:-devtest}
PORT=${CARLA_PORT:-2000}
TM_PORT=$((PORT + 6000))

SIMLINGO=/home/shamakg/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo
CHECKPOINT=$SIMLINGO/outputs/2026_05_30_23_47_08_parking_ft_v2/checkpoints/epoch=013.ckpt/last_fp32.pt
RESULT_DIR=/tmp/bench2drive_results
export SAVE_PATH=$RESULT_DIR/viz/
mkdir -p $RESULT_DIR $SAVE_PATH

cd $SIMLINGO

# agent_simlingo.py looks for conversation.py at relative pretrained/InternVL2-1B/
# from the working directory — symlink into place if missing
if [ ! -e pretrained ]; then
    ln -sf /home/shamakg/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/model/pretrained pretrained
fi

export CARLA_ROOT=/home/shamakg/opt/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:/home/shamakg/carla_garage/Bench2Drive/leaderboard
export PYTHONPATH=$PYTHONPATH:/home/shamakg/carla_garage/Bench2Drive/scenario_runner
export PYTHONPATH=$PYTHONPATH:$SIMLINGO
export PYTHONPATH=$PYTHONPATH:$SIMLINGO/team_code
export SCENARIO_RUNNER_ROOT=/home/shamakg/carla_garage/Bench2Drive/scenario_runner
export HUGGINGFACE_HUB_CACHE=/home/shamakg/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/model/pretrained
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source /home/shamakg/envs/simlingo/bin/activate

EVALUATOR=/home/shamakg/carla_garage/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py

run_routes() {
    local routes=$1
    local result=$2
    python -u $EVALUATOR \
        --routes=$routes \
        --repetitions=1 \
        --track=SENSORS \
        --checkpoint=$result \
        --timeout=600 \
        --agent=team_code/agent_simlingo.py \
        --agent-config=$CHECKPOINT \
        --traffic-manager-seed=1 \
        --port=$PORT \
        --traffic-manager-port=$TM_PORT
}

if [ "$MODE" = "devtest" ]; then
    echo "Running devtest (3 routes in Town12 — NOTE: may be OOD for base model)"
    run_routes \
        /home/shamakg/carla_garage/leaderboard/data/routes_devtest.xml \
        $RESULT_DIR/devtest_result.json

elif [ "$MODE" = "sample" ]; then
    echo "Running 5 bench2drive_split routes — actual simlingo eval distribution"
    SPLIT_DIR=/home/shamakg/carla_garage/leaderboard/data/bench2drive_split
    count=0
    for route_xml in $SPLIT_DIR/*.xml; do
        [ $count -ge 5 ] && break
        route_id=$(basename $route_xml .xml)
        result=$RESULT_DIR/${route_id}_result.json
        echo "--- $route_id ---"
        run_routes $route_xml $result || echo "FAILED: $route_id"
        count=$((count + 1))
    done
    echo "Results in $RESULT_DIR/"

elif [ "$MODE" = "full" ]; then
    echo "Running all bench2drive_split routes sequentially"
    SPLIT_DIR=/home/shamakg/carla_garage/leaderboard/data/bench2drive_split
    for route_xml in $SPLIT_DIR/*.xml; do
        route_id=$(basename $route_xml .xml)
        result=$RESULT_DIR/${route_id}_result.json
        echo "--- $route_id ---"
        run_routes $route_xml $result || echo "FAILED: $route_id"
    done
    echo "Results in $RESULT_DIR/"

else
    echo "Unknown mode: $MODE. Use 'devtest' or 'full'."
    exit 1
fi
