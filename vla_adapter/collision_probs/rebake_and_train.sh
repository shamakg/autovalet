#!/bin/bash
# Rebake run_topdown_001 (bicycle-cone window + vibrant green + un-masked risk_grid)
# then finetune with the ChauffeurNet-style collision loss enabled.
#
#   bash rebake_and_train.sh
#
# Re-baking regenerates topdown_heatmap/ (the model input) AND risk_grid/ (the
# loss target) from the recorded measurements -- no CARLA needed. Set REBAKE=0 to
# skip it on reruns once the grids exist.
set -euo pipefail

# ===================== TUNE ME (env-overridable) =============================
COLLISION_WEIGHT=${COLLISION_WEIGHT:-4.0}   # collision-loss weight (0=off). Higher
                                            # => avoids harder vs imitation. ~1-4.
TOPDOWN_HALF=${TOPDOWN_HALF:-18.0}          # must match config.TOPDOWN_HALF
REBAKE=${REBAKE:-1}                         # 1 = regen topdown_heatmap/+risk_grid/
                                            # (0 after a FRESH collection -- already baked)
REBUILD_BUCKETS=${REBUILD_BUCKETS:-1}       # 1 = rebuild buckets (single-pass)
# =============================================================================

ROOT=/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter
COLLISION_PROBS=${ROOT}/collision_probs
FINETUNE_DIR=${ROOT}/finetune
SIMLINGO=${ROOT}/simlingo
MODEL=${ROOT}/model

DATA_ROOT=${COLLISION_PROBS}/run_topdown_001/data/simlingo/parking_ft/routes_training/RouteScenario_parking
BUCKETS_V3=${COLLISION_PROBS}/parking_buckets_topdown_v3
PYTHON=/home/sumesh/envs/simlingo/bin/python

export PYTHONPATH="${SIMLINGO}:${SIMLINGO}/Bench2Drive/leaderboard/team_code:${COLLISION_PROBS}:${PYTHONPATH:-}"
export HUGGINGFACE_HUB_CACHE="${MODEL}/pretrained"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DIR=/home/sumesh/wandb_logs
export WANDB_ENTITY=shamakg-university-of-california-berkeley
mkdir -p "${WANDB_DIR}"

echo "=================================================================="
echo " collision_loss_weight = ${COLLISION_WEIGHT}   rebake=${REBAKE}   rebuild_buckets=${REBUILD_BUCKETS}"
echo "=================================================================="

# ── 0. rebake heatmaps + risk grids (cone window, vibrant green) ──────────────
if [ "${REBAKE}" = "1" ]; then
  echo "[0/2] Rebaking topdown_heatmap/ + risk_grid/ for all episodes (no CARLA)..."
  cd "${COLLISION_PROBS}"
  ${PYTHON} rebake.py --all --data-dir "${DATA_ROOT}"
fi

# ── 1. (re)build buckets via the single-pass builder ─────────────────────────
if [ "${REBUILD_BUCKETS}" = "1" ]; then
  echo "[1/2] Building buckets (single-pass; byte-identical to old v1+v3)..."
  ${PYTHON} "${FINETUNE_DIR}/create_parking_buckets_simple.py" \
      --data-dir "${DATA_ROOT}" \
      --out-dir  "${BUCKETS_V3}"
fi

# ── 2. train with the collision loss enabled ─────────────────────────────────
echo "[2/2] Training with collision_loss_weight=${COLLISION_WEIGHT}..."
cd "${SIMLINGO}"
${PYTHON} simlingo_training/train.py \
    --config-path ../../finetune \
    --config-name simlingo_seed1 \
    data_module.base_dataset.data_path=../collision_probs/run_topdown_001 \
    data_module.base_dataset.bucket_path=../collision_probs/parking_buckets_topdown_v3 \
    ++model.collision_loss_weight=${COLLISION_WEIGHT} \
    ++model.collision_topdown_half=${TOPDOWN_HALF}

echo "Done. Watch wandb for wps_collision_loss alongside speed_wps_loss."
