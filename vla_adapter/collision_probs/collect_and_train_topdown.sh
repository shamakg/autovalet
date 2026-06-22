#!/bin/bash
# collect_and_train_topdown.sh
#
# 1. Start CARLA
# 2. Collect topdown + heatmap data (collect_data_topdown.py)
# 3. Kill CARLA
# 4. Rebuild parking buckets
# 5. Train
#
# Usage:
#   bash collect_and_train_topdown.sh
#   COLLECT_EPISODES="pedestrian_normal:50,pedestrian_recovery:20" bash collect_and_train_topdown.sh
set -e

# ── paths ─────────────────────────────────────────────────────────────────────
COLLISION_PROBS=/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/collision_probs
FINETUNE_DIR=/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/finetune
OUTPUT_DIR=${OUTPUT_DIR:-${COLLISION_PROBS}/run_topdown_001}

# ── environment ───────────────────────────────────────────────────────────────
export CARLA_ROOT=/home/sumesh/opt/carla/PythonAPI/carla
export WORK_DIR=/home/sumesh/carla_garage
export CARLA_PORT=2000
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export SIMLINGO_ROOT=/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo
export SIMLINGO_TEAM_CODE=${SIMLINGO_ROOT}/team_code
export SIMLINGO_TRAINING=${SIMLINGO_ROOT}
export PYTHONPATH="${CARLA_ROOT}:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${SIMLINGO_TRAINING}:${SIMLINGO_TEAM_CODE}:${PYTHONPATH}"
export HOST=localhost
export OUTPUT_DIR

# source /opt/ros/humble/setup.bash
source /home/sumesh/envs/simlingo/bin/activate

PYTHON=/home/sumesh/envs/simlingo/bin/python

# ── CARLA cleanup (runs on exit for any reason) ───────────────────────────────
CARLA_PID=""
cleanup() {
    if [ -n "${CARLA_PID}" ]; then
        echo "[cleanup] Killing CARLA (PID ${CARLA_PID})..."
        kill "${CARLA_PID}" 2>/dev/null || true
        sleep 2
    fi
    pkill -f "CarlaUE4-Linux-Shipping" 2>/dev/null || true
    pkill -f "CarlaUE4" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── 1. start CARLA ────────────────────────────────────────────────────────────
echo "[1/5] Starting CARLA (port ${CARLA_PORT})..."
/home/sumesh/opt/carla/CarlaUE4.sh -RenderOffScreen -carla-port=${CARLA_PORT} &
CARLA_PID=$!
echo "      CARLA PID: ${CARLA_PID}"
echo "      Waiting 20s for CARLA to initialise..."
sleep 20

# ── 2. collect topdown + heatmap data ─────────────────────────────────────────
echo "[2/5] Collecting topdown + heatmap data -> ${OUTPUT_DIR}"
${PYTHON} "${COLLISION_PROBS}/collect_data_topdown.py"

# ── 3. kill CARLA ─────────────────────────────────────────────────────────────
echo "[3/5] Killing CARLA..."
CARLA_PID_SAVED=${CARLA_PID}
CARLA_PID=""   # prevent double-kill from trap
kill "${CARLA_PID_SAVED}" 2>/dev/null || true
sleep 5
pkill -f "CarlaUE4-Linux-Shipping" 2>/dev/null || true
pkill -f "CarlaUE4" 2>/dev/null || true
echo "      CARLA stopped."

# ── 4. build buckets + train with the collision loss ──────────────────────────
# Fresh collection already baked the cone heatmap + risk_grid/ (the BEV camera was
# present at collection time), so REBAKE=0. rebake_and_train.sh builds buckets with
# the single-pass builder and trains with collision_loss_weight enabled. Override
# the weight via env, e.g. COLLISION_WEIGHT=1.0 bash collect_and_train_topdown.sh
echo "[4/4] Building buckets + training (collision loss enabled)..."
REBAKE=0 bash "${COLLISION_PROBS}/rebake_and_train.sh"

echo "Done."
