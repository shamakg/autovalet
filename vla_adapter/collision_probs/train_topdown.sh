#!/bin/bash
# Train on run_topdown_001 data collected by collect_and_train_topdown.sh.
# Builds dedicated bucket files from the topdown data, then runs training.
set -e

COLLISION_PROBS=/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/collision_probs
FINETUNE_DIR=/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/finetune
SIMLINGO=/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo
MODEL=/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/model

DATA_DIR=${COLLISION_PROBS}/run_topdown_001/data/simlingo/parking_ft/routes_training/RouteScenario_parking
BUCKETS_V3=${COLLISION_PROBS}/parking_buckets_topdown_v3

export PYTHONPATH="${SIMLINGO}:${SIMLINGO}/Bench2Drive/leaderboard/team_code:${COLLISION_PROBS}:${PYTHONPATH}"
export HUGGINGFACE_HUB_CACHE="${MODEL}/pretrained"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DIR=/home/sumesh/wandb_logs
export WANDB_ENTITY=shamakg-university-of-california-berkeley
mkdir -p "${WANDB_DIR}"

PYTHON=/home/sumesh/envs/simlingo/bin/python

# ── 1. build buckets from topdown data (single-pass builder) ──────────────────
echo "[1/2] Building buckets from ${DATA_DIR}..."
${PYTHON} "${FINETUNE_DIR}/create_parking_buckets_simple.py" \
    --data-dir "${DATA_DIR}" \
    --out-dir  "${BUCKETS_V3}"
echo "Bucket distribution:"
cat "${BUCKETS_V3}/buckets_stats.json"
echo

# ── 2. train ──────────────────────────────────────────────────────────────────
echo "[2/2] Starting training..."
cd "${SIMLINGO}"


# /home/sumesh/envs/simlingo/bin/python simlingo_training/train.py \
#     --config-path ../../finetune \
#     --config-name simlingo_seed1 \
#     "data_module.base_dataset.data_path=../collision_probs/run_topdown_001" \
#     "data_module.base_dataset.bucket_path=../collision_probs/parking_buckets_topdown_v3"

# ── 2b. resume from a checkpoint ────────────────────────────────────────────────
# To RESUME a full training run (restores optimizer state, epoch, and wandb run),
# point resume_path at a Lightning .ckpt and set resume=true. Comment out the
# command above and use this instead:
#
RESUME_CKPT=/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo/outputs/2026_06_22_19_03_32_parking_ft_v2/checkpoints/last.ckpt
/home/sumesh/envs/simlingo/bin/python simlingo_training/train.py \
    --config-path ../../finetune \
    --config-name simlingo_seed1 \
    "data_module.base_dataset.data_path=../collision_probs/run_topdown_001" \
    "data_module.base_dataset.bucket_path=../collision_probs/parking_buckets_topdown_v3" \
    "resume_path=${RESUME_CKPT}" \
    "resume=true"
#
# Note: resume_path expects a full Lightning checkpoint (.ckpt, with trainer state).
# To instead only WARM-START the weights from a saved state_dict (no optimizer/epoch),
# set the `checkpoint=` override to a pytorch_model.pt file instead of resume_path.

echo "Done."
