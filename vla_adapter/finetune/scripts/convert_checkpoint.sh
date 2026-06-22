#!/bin/bash
# Convert a DeepSpeed sharded checkpoint to a single fp32 .pt file.
# Usage: convert_checkpoint.sh <epoch=NNN.ckpt directory>
set -e

CKPT_DIR="${1}"

if [ -z "${CKPT_DIR}" ]; then
    echo "Usage: $0 <path/to/epoch=NNN.ckpt>"
    exit 1
fi

CKPT_DIR="$(realpath ${CKPT_DIR})"
if [ ! -d "${CKPT_DIR}" ]; then
    echo "Error: ${CKPT_DIR} is not a directory"
    exit 1
fi

OUT_PT="${CKPT_DIR%.ckpt}_fp32.pt"

SIMLINGO=/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo
source /home/sumesh/envs/simlingo/bin/activate
export PYTHONPATH="${SIMLINGO}:${PYTHONPATH}"

echo "Converting ${CKPT_DIR} → ${OUT_PT}"
python "${CKPT_DIR}/zero_to_fp32.py" "${CKPT_DIR}" "${OUT_PT}"

# DeepSpeed sometimes writes a directory with pytorch_model.bin inside — flatten it
if [ -d "${OUT_PT}" ]; then
    echo "Flattening directory output..."
    mv "${OUT_PT}/pytorch_model.bin" /tmp/_ckpt_tmp.pt
    rmdir "${OUT_PT}"
    mv /tmp/_ckpt_tmp.pt "${OUT_PT}"
fi

echo "Done: $(ls -lh ${OUT_PT} | awk '{print $5, $9}')"
