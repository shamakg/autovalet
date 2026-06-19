"""Verify dataset_base._topdown_aug_affine against augment_route's transform.

The topdown BEV warp must move ground-plane content to exactly where the
shift-augmented waypoint labels expect it. We test that directly and purely
geometrically: place known obstacle points, render each at its real-pose pixel,
warp with the REAL helper, and confirm the warped blob lands where the same
rigid transform augment_route() applies (p_aug = R(yaw)^T @ (p - [0, y_trans]))
independently says it should -- in the same pixel<->ego convention the heatmap
is rendered with (crossing_gmm.route_corridor_mask).

Pass criterion: every point's warped centroid is within TOL_PX of its predicted
augmented pixel. A flipped rotation sign or swapped axis blows this up immediately.

Run from .../vla_adapter:  python finetune/tools/verify_topdown_aug.py
"""
import os
import sys

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_VLA_ADAPTER = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_VLA_ADAPTER, 'simlingo'))

from simlingo_training.dataloader.dataset_base import _topdown_aug_affine, _TOPDOWN_HALF_M

SIZE = 512
HALF = _TOPDOWN_HALF_M
TOL_PX = 1.5  # interpolation/centroid tolerance

# Ego points [x_forward, y_left] in metres, spread across the grid.
POINTS = [(8.0, 0.0), (5.0, 3.0), (12.0, -2.0), (3.0, 1.5), (6.0, -4.0)]
# (lateral shift m, yaw deg) — spans the training aug range (+/-1.5 m, +/-20 deg).
AUGS = [(0.0, 0.0), (1.5, 20.0), (-1.5, -15.0), (0.8, 5.0), (-1.0, -20.0)]


def ego_to_pixel(p):
    """Ego [x_fwd, y_left] -> pixel [col, row], matching route_corridor_mask."""
    m = SIZE / (2.0 * HALF)
    x, y = p
    return np.array([SIZE / 2.0 + m * y, SIZE / 2.0 - m * x])


def augment_point(p, y_trans, yaw_deg):
    """Replicate BaseDataset.augment_route: p_aug = R(yaw)^T @ (p - [0, y_trans])."""
    yaw = np.deg2rad(yaw_deg)
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return R.T @ (np.asarray(p, dtype=float) - np.array([0.0, y_trans]))


def blob_centroid(img):
    ys, xs = np.nonzero(img[:, :, 0])
    if len(xs) == 0:
        return None
    w = img[ys, xs, 0].astype(float)
    return np.array([(xs * w).sum() / w.sum(), (ys * w).sum() / w.sum()])  # [col, row]


def main():
    overall_max = 0.0
    failed = False
    for y_trans, yaw_deg in AUGS:
        M = _topdown_aug_affine(SIZE, y_trans, yaw_deg)
        errs = []
        for p in POINTS:
            src = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            px = ego_to_pixel(p)
            cv2.circle(src, (int(round(px[0])), int(round(px[1]))), 4, (255, 255, 255), -1)
            warped = cv2.warpAffine(src, M, (SIZE, SIZE), flags=cv2.INTER_LINEAR)
            got = blob_centroid(warped)
            if got is None:  # blob warped off-grid; compare against expected only if in-bounds
                exp = ego_to_pixel(augment_point(p, y_trans, yaw_deg))
                if 0 <= exp[0] < SIZE and 0 <= exp[1] < SIZE:
                    errs.append(1e9)
                continue
            exp = ego_to_pixel(augment_point(p, y_trans, yaw_deg))
            errs.append(float(np.linalg.norm(got - exp)))
        emax = max(errs) if errs else 0.0
        overall_max = max(overall_max, emax)
        status = "OK " if emax <= TOL_PX else "FAIL"
        if emax > TOL_PX:
            failed = True
        print(f"[{status}] y_trans={y_trans:+.1f}m yaw={yaw_deg:+5.1f}deg  max_err={emax:.3f}px")

    print(f"\nOverall max error: {overall_max:.3f}px (tol {TOL_PX}px)")
    print("RESULT:", "PASS" if not failed else "FAIL")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
