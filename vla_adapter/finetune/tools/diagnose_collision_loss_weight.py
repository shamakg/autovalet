"""Standalone diagnostic: load a finetuned checkpoint, run several validation
batches, and report the RAW (unweighted, weight=1.0) magnitude of the
collision-loss term next to route_loss/speed_wps_loss, so we can pick a
collision_loss_weight that keeps the collision term in the same order of
magnitude as the rest of the loss instead of dominating it.

This measures the trained checkpoint's actual predictions -- it does NOT
predict training dynamics under a different weight (a different weight
changes what the model learns to predict during training, not just the
loss scale at eval time). Treat this as "is weight=W in a sane ballpark",
not "weight=W will train exactly this well".

Usage:
    cd /home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/simlingo
    python ../finetune/tools/diagnose_collision_loss_weight.py <ckpt_path> [--num-batches N]
"""

import sys
import argparse
import statistics
from pathlib import Path

VLA_ADAPTER = Path(__file__).resolve().parents[2]
SIMLINGO = VLA_ADAPTER / "simlingo"
sys.path.insert(0, str(SIMLINGO))
sys.path.insert(0, str(SIMLINGO / "Bench2Drive" / "leaderboard" / "team_code"))

import torch
import hydra
import hydra.utils as _hydra_utils
from omegaconf import OmegaConf
from transformers import AutoProcessor
from pytorch_lightning.utilities import move_data_to_device

_hydra_utils.get_original_cwd = lambda: str(SIMLINGO)


def find_run_dir(ckpt_path: Path) -> Path:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=Path)
    parser.add_argument("--num-batches", type=int, default=20)
    args = parser.parse_args()

    ckpt_path = args.ckpt_path
    run_dir = find_run_dir(ckpt_path)
    cfg_path = run_dir / ".hydra" / "config.yaml"
    print(f"[diag] ckpt: {ckpt_path}")
    print(f"[diag] run_dir: {run_dir}")

    cfg = OmegaConf.load(cfg_path)
    cfg.gpus = 1
    cfg.data_module.batch_size = 4
    cfg.data_module.num_workers = 2
    print(f"[diag] configured collision_loss_weight: {cfg.model.get('collision_loss_weight', 0.0)}")

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
    try:
        val_loader = data_module.val_dataloader()
        split = "val"
    except Exception as e:
        print(f"[diag] val_dataloader unavailable ({e}), falling back to train_dataloader")
        val_loader = data_module.train_dataloader()
        split = "train"
    print(f"[diag] using {split} loader, {len(val_loader)} batches")

    print("[diag] instantiating model ...")
    model = hydra.utils.instantiate(
        cfg.model,
        cfg_data_module=cfg.data_module,
        processor=processor,
        cache_dir=None,
        _recursive_=False,
    )

    sd = load_state_dict(ckpt_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[diag] missing keys: {len(missing)}  unexpected keys: {len(unexpected)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # Force weight=1.0 so wps_collision_loss reports the RAW p-norm risk term;
    # this only rescales the logged number, it does NOT affect the model's
    # predictions (no backward pass here), so route_loss/speed_wps_loss in the
    # same forward are exactly what the checkpoint actually produces.
    orig_weight = model.adaptors.driving.collision_loss_weight
    model.adaptors.driving.collision_loss_weight = 1.0
    print(f"[diag] checkpoint trained with collision_loss_weight={orig_weight}, "
          f"running diagnostic with weight=1.0 to get raw magnitude")

    raw_collision, route_loss, speed_loss, n_with_risk_grid = [], [], [], 0

    it = iter(val_loader)
    with torch.no_grad():
        for i in range(args.num_batches):
            try:
                batch = next(it)
            except StopIteration:
                print(f"[diag] loader exhausted after {i} batches")
                break
            batch = move_data_to_device(batch, device)

            has_rg = (batch.driving_label is not None
                      and getattr(batch.driving_label, "risk_grid", None) is not None)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                output, _ = model.forward_loss(batch)
            avgs = output.loss_averages

            if "route_loss" in avgs:
                route_loss.append(avgs["route_loss"].item())
            if "speed_wps_loss" in avgs:
                speed_loss.append(avgs["speed_wps_loss"].item())
            if "wps_collision_loss" in avgs:
                raw_collision.append(avgs["wps_collision_loss"].item())
                n_with_risk_grid += 1
            other_keys = {k: v.item() for k, v in avgs.items()
                          if k not in ("route_loss", "speed_wps_loss", "wps_collision_loss")}
            print(f"[diag] batch {i}: has_risk_grid={has_rg} total_loss={output.loss.item():.4f} "
                  f"route_loss={avgs.get('route_loss', torch.tensor(float('nan'))).item():.4f} "
                  f"speed_wps_loss={avgs.get('speed_wps_loss', torch.tensor(float('nan'))).item():.4f} "
                  f"raw_collision={avgs.get('wps_collision_loss', torch.tensor(float('nan'))).item():.4f} "
                  f"other={other_keys}")

    model.adaptors.driving.collision_loss_weight = orig_weight

    def stats(xs, name):
        if not xs:
            print(f"[diag] {name}: no data")
            return
        print(f"[diag] {name}: n={len(xs)} mean={statistics.mean(xs):.4f} "
              f"median={statistics.median(xs):.4f} min={min(xs):.4f} max={max(xs):.4f}")

    print(f"\n========= SUMMARY ({n_with_risk_grid}/{len(route_loss) or 1} batches had risk_grid) =========")
    stats(route_loss, "route_loss")
    stats(speed_loss, "speed_wps_loss")
    stats(raw_collision, "raw_collision (weight=1.0)")

    if raw_collision and route_loss and speed_loss:
        other_mean = statistics.mean(route_loss) + statistics.mean(speed_loss)
        raw_mean = statistics.mean(raw_collision)
        print(f"\n[diag] route_loss + speed_wps_loss (mean) = {other_mean:.4f}")
        for w in (1, 2, 3, 4, 5):
            print(f"[diag] at collision_loss_weight={w}: "
                  f"wps_collision_loss={raw_mean * w:.4f}  "
                  f"(= {100*raw_mean*w/(other_mean + raw_mean*w):.1f}% of total loss)")


if __name__ == "__main__":
    main()
