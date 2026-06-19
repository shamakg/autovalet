#!/bin/bash
# Train on run_topdown_001 data collected by collect_and_train_topdown.sh.
# Builds dedicated bucket files from the topdown data, then runs training.
set -e

COLLISION_PROBS=/home/shamakg/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/collision_probs
FINETUNE_DIR=/home/shamakg/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/finetune
SIMLINGO=/home/shamakg/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo
MODEL=/home/shamakg/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/model

DATA_DIR=${COLLISION_PROBS}/run_topdown_001/data/simlingo/parking_ft/routes_training/RouteScenario_parking
BUCKETS_V1=${COLLISION_PROBS}/parking_buckets_topdown_v1
BUCKETS_V3=${COLLISION_PROBS}/parking_buckets_topdown_v3

export PYTHONPATH="${SIMLINGO}:${SIMLINGO}/Bench2Drive/leaderboard/team_code:${COLLISION_PROBS}:${PYTHONPATH}"
export HUGGINGFACE_HUB_CACHE="${MODEL}/pretrained"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DIR=/home/shamakg/wandb_logs
export WANDB_ENTITY=shamakg-university-of-california-berkeley
mkdir -p "${WANDB_DIR}"

PYTHON=/home/shamakg/envs/simlingo/bin/python

# ── 1. build v1 buckets from topdown data ─────────────────────────────────────
echo "[1/3] Building v1 buckets from ${DATA_DIR}..."
${PYTHON} "${FINETUNE_DIR}/create_parking_buckets.py" \
    --data-dir "${DATA_DIR}" \
    --out-dir  "${BUCKETS_V1}"

# ── 2. build v3 buckets (freeroll/ped/door splits) ────────────────────────────
echo "[2/3] Building v3 buckets..."
${PYTHON} "${FINETUNE_DIR}/create_parking_buckets_v3.py" \
    --v1-dir  "${BUCKETS_V1}" \
    --out-dir "${BUCKETS_V3}"
echo "Bucket distribution:"
cat "${BUCKETS_V3}/buckets_stats.json"
echo

# ── 3. train ──────────────────────────────────────────────────────────────────
echo "[3/3] Starting training..."
cd "${SIMLINGO}"

python simlingo_training/train.py \
    --config-path ../../finetune \
    --config-name simlingo_seed1 \
    "data_module.base_dataset.data_path=../collision_probs/run_topdown_001" \
    "data_module.base_dataset.bucket_path=../collision_probs/parking_buckets_topdown_v3"

echo "Done."
