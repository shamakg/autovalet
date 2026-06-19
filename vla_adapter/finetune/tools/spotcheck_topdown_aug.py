"""Visual spot-check: warp a REAL topdown_heatmap frame with the production
_topdown_aug_affine helper across several shift-augmentation values and dump a
labelled side-by-side composite so the geometry can be eyeballed on real data.

Picks the frame with the most coloured (non-grey) heatmap content so the corridor
and risk are clearly visible. Run from .../vla_adapter:
    python finetune/tools/spotcheck_topdown_aug.py
"""
import os
import sys
import glob

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_VLA_ADAPTER = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_VLA_ADAPTER, 'simlingo'))

from simlingo_training.dataloader.dataset_base import _topdown_aug_affine

_SCEN = os.path.join(
    _VLA_ADAPTER, 'collision_probs', 'run_topdown_001', 'data', 'simlingo',
    'parking_ft', 'routes_training', 'RouteScenario_parking', 'Town04_0196',
    'topdown_heatmap')
OUT_DIR = os.path.join(_VLA_ADAPTER, 'results', 'topdown_aug_spotcheck')

# (lateral shift m, yaw deg) — the no-op plus corners of the training aug range.
AUGS = [(0.0, 0.0), (1.5, 20.0), (-1.5, -20.0)]


def colourfulness(img_rgb):
    """Crude score: how much the pixel deviates from grey (heatmap tint present)."""
    f = img_rgb.astype(np.float32)
    return float(np.mean(np.abs(f - f.mean(axis=2, keepdims=True))))


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    # Allow an explicit frame path as argv[1]; otherwise auto-pick the most
    # colourful frame in the default scenario.
    if len(sys.argv) > 1:
        best = sys.argv[1]
        print(f"Selected frame (explicit): {best}")
    else:
        frames = sorted(glob.glob(os.path.join(_SCEN, '*.jpg')))
        if not frames:
            print(f"No frames under {_SCEN}")
            sys.exit(1)
        best, best_score = None, -1.0
        for p in frames[::3]:
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            sc = colourfulness(img)
            if sc > best_score:
                best, best_score = p, sc
        print(f"Selected frame: {best}  (colourfulness={best_score:.2f})")

    base = cv2.cvtColor(cv2.imread(best), cv2.COLOR_BGR2RGB)
    size = base.shape[0]

    tiles = []
    for y_trans, yaw_deg in AUGS:
        if y_trans == 0.0 and yaw_deg == 0.0:
            warped = base
            cap = "original (no aug)"
        else:
            M = _topdown_aug_affine(size, y_trans, yaw_deg)
            warped = cv2.warpAffine(base, M, (size, size), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
            cap = f"y={y_trans:+.1f}m  yaw={yaw_deg:+.0f}deg"
        tiles.append(label(warped, cap))

    composite = np.hstack(tiles)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'topdown_aug_composite.png')
    cv2.imwrite(out_path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
    # also dump the individual tiles
    for tile, (yt, yd) in zip(tiles, AUGS):
        name = 'original.png' if (yt == 0 and yd == 0) else f'aug_y{yt:+.1f}_yaw{yd:+.0f}.png'
        cv2.imwrite(os.path.join(OUT_DIR, name), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))

    print(f"\nComposite : {out_path}")
    print(f"Tiles dir : {OUT_DIR}")


if __name__ == "__main__":
    main()
