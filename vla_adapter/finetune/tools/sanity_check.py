"""
Sanity check: load a checkpoint, run a few val batches, save waypoint visualizations.
Must be run via scripts/sanity_check.sh (which sets PYTHONPATH and cd's to simlingo/).

Usage (from simlingo/ dir):
    python ../finetune/sanity_check.py experiment=parking_ft \
        data_module.num_workers=0 \
        data_module.batch_size=4 \
        checkpoint=outputs/2026_05_24_23_44_22_parking_ft_v2/checkpoints/epoch=004.ckpt

Plots saved to /tmp/sanity_check/.
"""

import os
from pathlib import Path

import hydra
from simlingo_training.config import TrainConfig  # registers structured configs with Hydra ConfigStore
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from omegaconf import DictConfig
from pytorch_lightning.utilities import move_data_to_device
from transformers import AutoProcessor

NUM_BATCHES = 3
OUTPUT_DIR = Path("/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/finetune/outputs/sanity_check")
# Update this path to point to the checkpoint you want to test
CHECKPOINT = "/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo/outputs/2026_05_24_23_44_22_parking_ft_v2/checkpoints/epoch=004.ckpt"


@hydra.main(config_path="../simlingo/simlingo_training/config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(cfg.model.vision_model.variant, trust_remote_code=True)

    print("Setting up datamodule...")
    data_module = hydra.utils.instantiate(
        cfg.data_module,
        processor=processor,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant=cfg.model.language_model.variant,
        _recursive_=False,
    )
    data_module.setup(stage=None)

    print("Instantiating model...")
    model = hydra.utils.instantiate(
        cfg.model,
        cfg_data_module=cfg.data_module,
        processor=processor,
        cache_dir=None,
        _recursive_=False,
    )

    print(f"Loading checkpoint: {CHECKPOINT}")
    if os.path.isdir(CHECKPOINT):
        state_dict = get_fp32_state_dict_from_zero_checkpoint(CHECKPOINT)
    else:
        state_dict = torch.load(CHECKPOINT, map_location="cpu")
    model.load_state_dict(state_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"Model loaded on {device}\n")

    wp_errors, route_errors = [], []

    for i, batch in enumerate(data_module.val_dataloader()):
        if i >= NUM_BATCHES:
            break

        batch = move_data_to_device(batch, device)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
            speed_wps, route, language = model(batch, return_language=True)

        gt_wps = batch.driving_label.waypoints[:, :11].cpu().float()
        pred_wps = speed_wps[:, :11].cpu().float()
        wp_err = (pred_wps - gt_wps).norm(dim=-1).mean().item()
        wp_errors.append(wp_err)

        gt_route = batch.driving_label.path[:, :20].cpu().float()
        pred_route = route[:, :20].cpu().float()
        route_err = (pred_route - gt_route).norm(dim=-1).mean().item()
        route_errors.append(route_err)

        print(f"Batch {i+1}/{NUM_BATCHES}  |  waypoint L2: {wp_err:.4f}m  |  route L2: {route_err:.4f}m")
        if language:
            print(f"  language[0]: {language[0][:120]}")

        _save_plot(batch, pred_wps, gt_wps, pred_route, gt_route, i)

    print(f"\n=== Summary ===")
    print(f"Mean waypoint L2 error : {np.mean(wp_errors):.4f}m")
    print(f"Mean route L2 error    : {np.mean(route_errors):.4f}m")
    print(f"Plots saved to {OUTPUT_DIR}/")


def _save_plot(batch, pred_wps, gt_wps, pred_route, gt_route, batch_idx):
    B = min(pred_wps.shape[0], 4)
    pred_wps = pred_wps[:B].numpy()
    gt_wps = gt_wps[:B].numpy()
    pred_route = pred_route[:B].numpy()
    gt_route = gt_route[:B].numpy()

    fig, axes = plt.subplots(2, B, figsize=(4 * B, 8))
    if B == 1:
        axes = axes.reshape(2, 1)

    for i in range(B):
        for row, (pred, gt, title) in enumerate([
            (pred_wps[i], gt_wps[i], "Speed WPs"),
            (pred_route[i], gt_route[i], "Route"),
        ]):
            ax = axes[row, i]
            ax.plot(pred[:, 1], pred[:, 0], "b-o", label="pred", markersize=4)
            ax.plot(gt[:, 1], gt[:, 0], "g-x", label="GT", markersize=6)
            ax.set_title(f"{title} [{i}]")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True)
            if i == 0:
                ax.legend(fontsize=8)

    plt.suptitle(f"Batch {batch_idx} — blue=pred, green=GT")
    plt.tight_layout()
    out = OUTPUT_DIR / f"batch{batch_idx}.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
