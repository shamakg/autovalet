"""Standalone diagnostic: load a finetuned checkpoint and one training sample, run
a forward pass, print predicted route + speed_wps next to the ground-truth labels.

Usage:
    cd /home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo
    python ../finetune/diagnose_ckpt.py <ckpt_path>

ckpt_path can be either a flat .pt file or a deepspeed checkpoint directory.
"""

import os
import sys
import argparse
from pathlib import Path

VLA_ADAPTER = Path(__file__).resolve().parents[1]
SIMLINGO = VLA_ADAPTER / "simlingo"
sys.path.insert(0, str(SIMLINGO))
sys.path.insert(0, str(SIMLINGO / "Bench2Drive" / "leaderboard" / "team_code"))

import torch
import hydra
import hydra.utils as _hydra_utils
from omegaconf import OmegaConf
from transformers import AutoProcessor

# Hydra's get_original_cwd() is only valid inside @hydra.main; the dataloader
# calls it to anchor relative paths. We run the diagnostic from the simlingo
# directory, which is what get_original_cwd() returns during normal training.
_hydra_utils.get_original_cwd = lambda: str(SIMLINGO)


def find_run_dir(ckpt_path: Path) -> Path:
    """Walk up from the ckpt to the run dir (the one that has .hydra/)."""
    p = ckpt_path.resolve()
    for parent in [p] + list(p.parents):
        if (parent / ".hydra" / "config.yaml").is_file():
            return parent
    raise FileNotFoundError(f"No .hydra/config.yaml found above {ckpt_path}")


def load_state_dict(ckpt_path: Path):
    if ckpt_path.is_dir():
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        return get_fp32_state_dict_from_zero_checkpoint(str(ckpt_path))
    return torch.load(str(ckpt_path), map_location="cpu")


def fmt(t):
    if t is None:
        return "None"
    if isinstance(t, torch.Tensor):
        return f"shape={tuple(t.shape)} dtype={t.dtype} norm={t.norm().item():.3f} first5={t.flatten()[:5].tolist()}"
    return repr(t)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=Path)
    parser.add_argument("--num-samples", type=int, default=1)
    args = parser.parse_args()

    ckpt_path = args.ckpt_path
    run_dir = find_run_dir(ckpt_path)
    cfg_path = run_dir / ".hydra" / "config.yaml"
    print(f"[diag] ckpt: {ckpt_path}")
    print(f"[diag] run_dir: {run_dir}")
    print(f"[diag] cfg: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)
    cfg.gpus = 1
    cfg.data_module.batch_size = 1
    cfg.data_module.num_workers = 0
    # Force the dataloader to use the parking data so we can diagnose any
    # checkpoint on the same samples (e.g. pretrained vs finetuned).
    cfg.data_module.base_dataset.data_path = "../finetune/run_001"
    cfg.data_module.base_dataset.bucket_path = "database/bucketsv2_simlingo"
    cfg.data_module.base_dataset.use_old_towns = True
    cfg.data_module.base_dataset.use_town13 = True
    # carla_no_buckets bypasses buckets via bucket_name="all" — make sure it's set.
    if hasattr(cfg.data_module, 'train_partitions'):
        cfg.data_module.train_partitions = None
    if hasattr(cfg.data_module, 'train_partitions_dreamer'):
        cfg.data_module.train_partitions_dreamer = None

    print(f"[diag] vision_model: {cfg.model.vision_model.variant}")
    print(f"[diag] route_as: {cfg.data_module.base_dataset.route_as}")
    print(f"[diag] pred_len: {cfg.data_module.base_dataset.pred_len}")

    processor = AutoProcessor.from_pretrained(cfg.model.vision_model.variant, trust_remote_code=True)

    print("[diag] instantiating data_module ...")
    data_module = hydra.utils.instantiate(
        cfg.data_module,
        processor=processor,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant=cfg.model.language_model.variant,
        _recursive_=False,
    )
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()
    print(f"[diag] train_loader produced {len(train_loader)} batches")

    print("[diag] instantiating model ...")
    model = hydra.utils.instantiate(
        cfg.model,
        cfg_data_module=cfg.data_module,
        processor=processor,
        cache_dir=None,
        _recursive_=False,
    )

    print(f"[diag] loading state_dict from {ckpt_path}")
    sd = load_state_dict(ckpt_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[diag] missing keys: {len(missing)}  unexpected keys: {len(unexpected)}")
    if missing:
        print(f"[diag]   first missing: {missing[:5]}")
    if unexpected:
        print(f"[diag]   first unexpected: {unexpected[:5]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    it = iter(train_loader)
    with torch.no_grad():
        for s in range(args.num_samples):
            batch = next(it)

            def to_dev(x):
                if isinstance(x, torch.Tensor):
                    return x.to(device)
                return x

            from dataclasses import is_dataclass, fields
            def move(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.to(device)
                if is_dataclass(obj):
                    return obj.__class__(**{f.name: move(getattr(obj, f.name)) for f in fields(obj)})
                if hasattr(obj, "_fields"):  # NamedTuple
                    return obj.__class__(*[move(getattr(obj, f)) for f in obj._fields])
                return obj

            batch = move(batch)

            print(f"\n========= SAMPLE {s} =========")
            speed_wps, route, language = model(batch, return_language=False)

            label_path = batch.driving_label.path if batch.driving_label is not None else None
            label_waypoints = batch.driving_label.waypoints if batch.driving_label is not None else None

            print(f"[pred] route       {fmt(route)}")
            print(f"[gt]   path        {fmt(label_path)}")
            print(f"[pred] speed_wps   {fmt(speed_wps)}")
            print(f"[gt]   waypoints   {fmt(label_waypoints)}")

            if route is not None and label_path is not None and route.shape == label_path.shape:
                diff = (route - label_path).abs().mean().item()
                print(f"[diff] mean |pred - gt| route: {diff:.4f}")
                # signed mean of dy per waypoint: positive = consistent left bias, negative = consistent right
                y_bias = (route[..., 1] - label_path[..., 1]).mean().item()
                x_bias = (route[..., 0] - label_path[..., 0]).mean().item()
                print(f"[bias] route x_bias={x_bias:+.4f}  y_bias={y_bias:+.4f}")
            if speed_wps is not None and label_waypoints is not None:
                k = min(speed_wps.shape[1], label_waypoints.shape[1])
                diff = (speed_wps[:, :k] - label_waypoints[:, :k]).abs().mean().item()
                print(f"[diff] mean |pred - gt| speed_wps (first {k}): {diff:.4f}")
                y_bias = (speed_wps[:, :k, 1] - label_waypoints[:, :k, 1]).mean().item()
                x_bias = (speed_wps[:, :k, 0] - label_waypoints[:, :k, 0]).mean().item()
                print(f"[bias] speed_wps x_bias={x_bias:+.4f}  y_bias={y_bias:+.4f}")


if __name__ == "__main__":
    main()
