#!/bin/bash
set -e

FINETUNE_DIR=/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/finetune
DATA_DIR=${FINETUNE_DIR}/run_001/data

# ── environment ──────────────────────────────────────────────────────────────
export CARLA_ROOT=/home/sumesh/opt/carla/PythonAPI/carla
export WORK_DIR=/home/sumesh/carla_garage
export CARLA_PORT=2000
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export SIMLINGO_ROOT=/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo
export SIMLINGO_TEAM_CODE=${SIMLINGO_ROOT}/team_code
export SIMLINGO_TRAINING=${SIMLINGO_ROOT}
export PYTHONPATH="${CARLA_ROOT}:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${SIMLINGO_TRAINING}:${SIMLINGO_TEAM_CODE}:${PYTHONPATH}"
source /opt/ros/humble/setup.bash
source /home/sumesh/envs/simlingo/bin/activate
export HOST=localhost

PYTHON=/home/sumesh/envs/simlingo/bin/python

# ── CARLA cleanup (runs on exit for any reason) ───────────────────────────────
CARLA_PID=""
cleanup() {
    if [ -n "${CARLA_PID}" ]; then
        echo "[cleanup] Killing CARLA (PID ${CARLA_PID})..."
        kill ${CARLA_PID} 2>/dev/null || true
        sleep 2
        pkill -f "CarlaUE4-Linux-Shipping" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ── 1. clear old data ────────────────────────────────────────────────────────
echo "[1/6] Removing old episode data..."
rm -rf "${DATA_DIR}"

# ── 2. start CARLA ───────────────────────────────────────────────────────────
echo "[2/6] Starting CARLA (port ${CARLA_PORT})..."
/home/sumesh/opt/carla/CarlaUE4.sh -RenderOffScreen -carla-port=${CARLA_PORT} &
CARLA_PID=$!
echo "      CARLA PID: ${CARLA_PID}"
echo "      Waiting 20s for CARLA to be ready..."
sleep 20

# ── 3. collect data ──────────────────────────────────────────────────────────
echo "[3/6] Collecting data..."
${PYTHON} ${FINETUNE_DIR}/collect_data.py

# ── 4. shut down CARLA (trap will also run on normal exit, this is explicit) ─
echo "[4/6] CARLA will be shut down by cleanup trap."
CARLA_PID_SAVED=${CARLA_PID}
CARLA_PID=""   # prevent double-kill from trap after we kill manually here
kill ${CARLA_PID_SAVED} 2>/dev/null || true
sleep 5
pkill -f "CarlaUE4-Linux-Shipping" 2>/dev/null || true

# ── 5. rebuild sampling buckets from the freshly collected data ──────────────
# buckets_paths.pkl is a hard-coded list of frame paths into run_001/data, which
# we just deleted + recollected. It MUST be rebuilt (v1 -> v3) or training samples
# stale/missing frames. Builder defaults already target run_001/data ->
# parking_buckets_v1 -> parking_buckets_v3 (the path train.sh's config reads).
echo "[5/6] Rebuilding parking buckets (v1 -> v3)..."
${PYTHON} ${FINETUNE_DIR}/create_parking_buckets.py
${PYTHON} ${FINETUNE_DIR}/create_parking_buckets_v3.py
echo "      New bucket distribution (the 0.5 m/s crawl shifts frames into the"
echo "      near-spot buckets — eyeball the 2:1 freeroll:ped ratios before the long train):"
cat ${FINETUNE_DIR}/parking_buckets_v3/buckets_stats.json
echo

# ── 6. train ─────────────────────────────────────────────────────────────────
echo "[6/6] Starting training..."
bash ${FINETUNE_DIR}/scripts/train.sh

echo "Done."


### AFTER TRAINING, 