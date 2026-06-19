"""Quick test: opposite-vehicle scenario heatmap (no CARLA needed)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from crossing_gmm import render_ego_bev_image
from PIL import Image
import numpy as np

# Ego at origin, yaw=0 (facing +x). Oncoming vehicle is 15m ahead,
# travelling directly toward ego along -x (cross_dir = [-1, 0]).
# Vehicle footprint: half_length=2.5m, half_width=1.0m.
vehicles = [{
    'edge_point':     [15.0, 0.5],   # 15m ahead, slightly to the left
    'cross_dir':      [-1.0, 0.0],   # coming straight at us
    'cross_distance': 20.0,          # enough runway to sweep across
    'speed':          8.0,           # ~30 km/h
    'p_cross':        1.0,           # certainty — it will reach us
    'extent':         [2.5, 1.0],    # sedan footprint
}]

HALF = 18.0
img = render_ego_bev_image(
    vehicles, ego_xy=[0.0, 0.0], ego_yaw=0.0,
    x_range=(-HALF, HALF), y_range=(-HALF, HALF),
    resolution=0.5,
    colormap=True, soft_colors=True,
    horizon_scale=1.5,
)
out = pathlib.Path(__file__).parent / "vehicle_heatmap_test.png"
Image.fromarray(img).save(out)
print(f"Saved {out}  shape={img.shape}  max={img.max()}")
