#!/bin/bash
SIMLINGO=/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo

export PYTHONPATH="${SIMLINGO}:${SIMLINGO}/Bench2Drive/leaderboard/team_code:${PYTHONPATH}"

MODEL=/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/model
export HUGGINGFACE_HUB_CACHE="${MODEL}/pretrained"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export WANDB_DIR=/home/sumesh/wandb_logs
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
#     checkpoint=/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo/outputs/2026_05_23_10_15_34_parking_ft_v2/checkpoints/epoch003_fp32.pt

# python simlingo_training/train.py \
#       --config-path ../../finetune \
#       --config-name simlingo_seed1 \
#       resume=true \
#       resume_path=/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo/outputs/2026_05_28_17_25_46_parking_ft_v2/checkpoints/last.ckpt \
#       checkpoint=null \
#       wandb_name=2026_05_28_17_25_46