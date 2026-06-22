#!/usr/bin/env python3
"""
Evaluate a trained checkpoint on held-out destination parking spots.

Run prepare_destination_split.py first to create the split directory, then:

    python finetune/scripts/eval_destinations.py \
        --checkpoint outputs/2026_05_23_.../checkpoints/epoch003_fp32.pt \
        --split-dir  finetune/run_001_dest_split

This script reuses the simlingo training infrastructure (no code modified)
and runs one validation pass — no gradient updates, just loss computation.

Must be run from the simlingo directory (same as train.sh):
    cd /home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo
    python ../finetune/scripts/eval_destinations.py ...
"""

import argparse
import os
import sys

# ── PYTHONPATH — mirror what train.sh does ────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_VLA_DIR    = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_SIMLINGO   = os.path.join(_VLA_DIR, "simlingo")
sys.path.insert(0, _SIMLINGO)
sys.path.insert(0, os.path.join(_SIMLINGO, "Bench2Drive", "leaderboard", "team_code"))


def main():
    parser = argparse.ArgumentParser(
        description="Run validation-only on held-out destination episodes."
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to a .pt state-dict checkpoint (e.g. epoch003_fp32.pt) or .ckpt file",
    )
    parser.add_argument(
        "--split-dir", required=True,
        help="Path to the dest-split root dir created by prepare_destination_split.py",
    )
    parser.add_argument(
        "--precision", default="16-mixed",
        help="Trainer precision (default: 16-mixed)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
    )
    parser.add_argument(
        "--output-dir", default="eval_results",
        help="Where to write the CSV results (default: eval_results/)",
    )
    args = parser.parse_args()

    checkpoint = os.path.abspath(args.checkpoint)
    split_dir  = os.path.abspath(args.split_dir)

    if not os.path.exists(checkpoint):
        sys.exit(f"Checkpoint not found: {checkpoint}")
    if not os.path.isdir(split_dir):
        sys.exit(f"Split directory not found: {split_dir}\n"
                 f"Run prepare_destination_split.py first.")

    # Compute data_path relative to the simlingo directory (matches how parking_ft.yaml works)
    data_path = os.path.relpath(split_dir, _SIMLINGO)

    # ── disable wandb ──────────────────────────────────────────────────────────
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLE_CODE"] = "True"

    # ── load config via Hydra ──────────────────────────────────────────────────
    import torch
    import pytorch_lightning as pl
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    from transformers import AutoProcessor

    config_dir = os.path.join(_SIMLINGO, "simlingo_training", "config")
    GlobalHydra.instance().clear()

    overrides = [
        "experiment=parking_ft",
        f"data_module.base_dataset.data_path={data_path}",
        "data_module.base_dataset.use_town13=false",
        f"data_module.batch_size={args.batch_size}",
        f"data_module.num_workers={args.num_workers}",
        "debug=true",        # keeps wandb offline
        "val_every_n_epochs=1",
    ]

    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = compose(config_name="config", overrides=overrides)

    print("Config overrides applied:")
    print(f"  data_path      = {data_path}")
    print(f"  use_town13     = false")
    print(f"  checkpoint     = {checkpoint}")
    print()

    # ── build model + data module (mirrors train.py) ───────────────────────────
    import hydra

    processor = AutoProcessor.from_pretrained(
        cfg.model.vision_model.variant, trust_remote_code=True
    )

    data_module = hydra.utils.instantiate(
        cfg.data_module,
        processor=processor,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant=cfg.model.language_model.variant,
        _recursive_=False,
    )

    model = hydra.utils.instantiate(
        cfg.model,
        cfg_data_module=cfg.data_module,
        processor=processor,
        cache_dir=None,
        _recursive_=False,
    )

    # ── load checkpoint ────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {checkpoint}")
    if checkpoint.endswith(".ckpt"):
        # Lightning checkpoint — extract state dict
        ckpt = torch.load(checkpoint, map_location="cpu")
        state_dict = ckpt["state_dict"]
        # Lightning prepends 'model.' to keys in some versions
        if all(k.startswith("model.") for k in state_dict):
            state_dict = {k[len("model."):]: v for k, v in state_dict.items()}
    else:
        # Raw fp32 state dict (e.g. epoch003_fp32.pt)
        state_dict = torch.load(checkpoint, map_location="cpu")

    model.load_state_dict(state_dict)
    print("Checkpoint loaded.\n")

    # ── run validation only ────────────────────────────────────────────────────
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import CSVLogger

    os.makedirs(args.output_dir, exist_ok=True)
    csv_logger = CSVLogger(args.output_dir, name="val_destinations")

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision=args.precision,
        logger=csv_logger,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    print("Running validation on held-out destination episodes...")
    results = trainer.validate(model, datamodule=data_module, verbose=True)

    print("\n" + "="*50)
    print("Validation results:")
    for k, v in results[0].items():
        print(f"  {k:40s} {v:.6f}")

    out_csv = os.path.join(args.output_dir, "val_destinations", "version_0", "metrics.csv")
    if os.path.exists(out_csv):
        print(f"\nFull metrics written to: {out_csv}")

    print("="*50)


if __name__ == "__main__":
    main()
