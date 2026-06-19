#!/bin/bash
SIMLINGO=/home/shamakg/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo

export PYTHONPATH="${SIMLINGO}:${SIMLINGO}/Bench2Drive/leaderboard/team_code:${PYTHONPATH}"

MODEL=/home/shamakg/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/model
export HUGGINGFACE_HUB_CACHE="${MODEL}/pretrained"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export WANDB_DIR=/home/shamakg/wandb_logs
export WANDB_ENTITY=shamakg-university-of-california-berkeley
mkdir -p "${WANDB_DIR}"

cd ${SIMLINGO}

python simlingo_training/train.py \
    --config-path ../../finetune \
    --config-name simlingo_seed1

# python simlingo_training/train.py experiment=parking_ft \
#     max_epochs=10 \
#     data_module.num_workers=4 \
#     data_module.batch_size=4 \
#     precision=16-mixed \
#     val_every_n_epochs=1 \
#     resume=false \
#     debug=false \
#     checkpoint=/home/shamakg/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo/outputs/2026_05_23_10_15_34_parking_ft_v2/checkpoints/epoch003_fp32.pt

# WANDB_ENTITY=njain110706-university-of-california-berkeley python simlingo_training/train.py \
#   --config-path ../../finetune --config-name simlingo_seed1 \
#   data_module.base_dataset.data_path=../collision_probs/run_topdown_001 \
#   data_module.base_dataset.bucket_path=../collision_probs/parking_buckets_topdown_v3 \
#   resume=True \
#   resume_path=/home/shamakg/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo/outputs/2026_06_18_10_51_40_parking_ft_v2/checkpoints/last.ckpt