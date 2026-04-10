#!/bin/bash

# Re-download SimLingo checkpoint
HF_HOME=/tmp/hf_cache huggingface-cli download RenzKa/simlingo \
    simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt \
    --local-dir /tmp/simlingo_download/

# Set up checkpoint directory structure
mkdir -p /tmp/simlingo_ckpt/checkpoints/epoch=013.ckpt
ln -sf /tmp/simlingo_download/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt \
    /tmp/simlingo_ckpt/checkpoints/epoch=013.ckpt/pytorch_model.pt

# Download hydra config
HF_HOME=/tmp/hf_cache huggingface-cli download RenzKa/simlingo \
    simlingo/.hydra/config.yaml \
    --local-dir /tmp/simlingo_download/

mkdir -p /tmp/simlingo_ckpt/.hydra
cp /tmp/simlingo_download/simlingo/.hydra/config.yaml /tmp/simlingo_ckpt/.hydra/

# Download InternVL2-1B
HF_HOME=/tmp/hf_cache huggingface-cli download OpenGVLab/InternVL2-1B \
    model.safetensors \
    --local-dir /tmp/pretrained/InternVL2-1B/

# Fix HF hub cache symlink
mkdir -p /tmp/hf_cache/hub/models--OpenGVLab--InternVL2-1B/snapshots/0d75ccd166b1d0b79446ae6c5d1a4a667f1e6187
ln -sf /tmp/pretrained/InternVL2-1B /tmp/hf_cache/hub/models--OpenGVLab--InternVL2-1B/snapshots/0d75ccd166b1d0b79446ae6c5d1a4a667f1e6187

echo "Done!"