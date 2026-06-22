#!/bin/bash
set -e

echo "[after_collect] Waiting for collect_data.py to finish..."
while pgrep -f "collect_data.py" > /dev/null; do
    sleep 30
done
echo "[after_collect] collect_data.py done."

echo "[after_collect] Killing CARLA..."
pkill -f "CarlaUE4" || true
pkill -f "CarlaUnreal" || true
sleep 5
echo "[after_collect] CARLA killed."

echo "[after_collect] Starting training..."
bash /home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/finetune/scripts/train.sh
echo "[after_collect] Training done."
