
import sys, os, torch, numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
_AUTOVALET_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _AUTOVALET_ROOT)
sys.path.insert(0, "/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/diffusion_adapter/nuplan-devkit")

_DP_ROOT = os.path.join(_HERE, "Diffusion-Planner")
if _DP_ROOT not in sys.path:
    sys.path.insert(0, _DP_ROOT)

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
import diff_adapter as da
from diffusion_planner.data_process.utils import (
    convert_to_model_inputs,
    _state_se2_array_to_transform_matrix,
    _state_se2_array_to_transform_matrix_batch,
    _transform_matrix_to_state_se2_array_batch,
)
from utils.coord_utils import carla_transform_to_standard, standard_to_carla
from utils.map_process import LANE_NUM, ROUTE_LANE_NUM, LANE_LEN
from diffusion_planner.utils.config import Config
from diffusion_planner.data_process.utils import convert_to_model_inputs
from utils.map_process import LANE_NUM, build_map_features

CKPT = "/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/diffusion_adapter/checkpoints/model.pth"   # fill in
ARGS = "/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/diffusion_adapter/checkpoints/args.json"

config = Config(ARGS, guidance_fn=None)
da._config = config

anchor = np.array(carla_transform_to_standard(285.6, -243.7, 90.0))
da._start_std = anchor[:2]
da._dest_std = np.array([285.6, -183.7])  # straight ahead

route = da._make_straight_route_lanes(anchor)
print("Route lane 0, first 3 pts (before norm):")
print(route[0, :3, :4])

data = {
    'neighbor_agents_past':        np.zeros((32,21,11), dtype=np.float32),
    'ego_current_state':           np.array([0,0,1,0,2,0,0,0,0,0], dtype=np.float32),
    'static_objects':              np.zeros((5,10), dtype=np.float32),
    'lanes':                       np.zeros((LANE_NUM,20,12), dtype=np.float32),
    'lanes_speed_limit':           np.full((LANE_NUM,1), 14., dtype=np.float32),
    'lanes_has_speed_limit':       np.ones((LANE_NUM,1), dtype=np.bool_),
    'route_lanes':                 route,
    'route_lanes_speed_limit':     np.full((config.route_num,1), 14., dtype=np.float32),
    'route_lanes_has_speed_limit': np.ones((config.route_num,1), dtype=np.bool_),
}
# Before normalization
print("Raw Ego vs Route Start:", data['ego_current_state'][0], route[0,0,0])
inputs = convert_to_model_inputs(data, 'cuda')
normed = config.observation_normalizer(inputs)


r = normed['route_lanes'][0]  # (route_num, route_len, 12)
nonzero = (r.abs().sum(dim=-1) > 0).sum().item()
print("Norm Ego vs Route Start:", normed['ego_current_state'][0,0].item(), r[0,0,0].item())

print(f"Non-zero route lane points after norm: {nonzero}/{config.route_num * config.route_len}")
print("Lane 0, pt 0 after norm:", r[0, 0].cpu().numpy()[:4])
print("Lane 0, pt 1 after norm:", r[0, 1].cpu().numpy()[:4])

# --- User requested tests for coord_utils ---
print("\n--- Testing coord_utils.py ---")
# The current coord_utils is a simple pass-through for x and y, and np.deg2rad for heading.
# Standard: x=x, y=y, heading=np.deg2rad(heading_deg)
cx, cy, ch = 10.0, 5.0, 90.0
sx, sy, sh = carla_transform_to_standard(cx, cy, ch)
print(f"CARLA (x={cx}, y={cy}, h={ch} deg) -> Standard (x={sx}, y={sy}, h={sh:.4f} rad)")
assert sx == cx and sy == cy, "Standard x,y should match CARLA x,y based on current coord_utils"

vx, vy = 5.0, -2.0
svx, svy = da.carla_velocity_to_standard(vx, vy) if hasattr(da, 'carla_velocity_to_standard') else (vx, vy)
print(f"CARLA vel (vx={vx}, vy={vy}) -> Standard vel (vx={svx}, vy={svy})")

rcx, rcy, rch = standard_to_carla(sx, sy, sh)
print(f"Standard (x={sx}, y={sy}, h={sh:.4f} rad) -> CARLA (x={rcx}, y={rcy}, h={rch} deg)")
assert rcx == cx and rcy == cy and np.isclose(rch, ch), "Roundtrip conversion failed"

print("coord_utils tests passed! Coordinates are behaving exactly as currently defined.\n")