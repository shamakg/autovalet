#!/bin/bash
source /home/sumesh/envs/simlingo/bin/activate

SIMLINGO=/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo

export PYTHONPATH="${SIMLINGO}:${SIMLINGO}/Bench2Drive/leaderboard/team_code:${PYTHONPATH}"

MODEL=/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/model
export HUGGINGFACE_HUB_CACHE="${MODEL}/pretrained"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd ${SIMLINGO}

python ../finetune/sanity_check.py experiment=parking_ft \
    data_module.num_workers=0 \
    data_module.batch_size=4
