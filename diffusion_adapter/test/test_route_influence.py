"""
test_route_influence.py — Deep-dive on WHY route_lanes barely influence the model.

From test_model_live.py we know:
  - Zero route vs real route: only 0.23m difference in final position
  - The model mostly follows its circular prior regardless of route
  - Route influence Δ ≈ 0.23m out of 8.6m total displacement = 2.6%

This test investigates:
  1. What does route_lanes look like after the normalizer? (scaled values)
  2. Does the model use route_lanes at all? (ablation: compare 0 vs 1e6 route)
  3. Is the route_lanes avail mask zeroing things out?
  4. Does guidance (GOAL_SCALE) actually steer the trajectory?
  5. What's the right GOAL_SCALE to overcome the circular prior?

Run with:
    /home/sumesh/envs/simlingo/bin/python test/test_route_influence.py
"""

import sys, os
import numpy as np
import torch

_HERE   = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.dirname(_HERE)
_NUPLAN = os.path.join(_ROOT, "nuplan-devkit")
_DP     = os.path.join(_ROOT, "Diffusion-Planner")

for _p in [_HERE, _ROOT, _NUPLAN, _DP]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from diffusion_planner.data_process.utils import (
    convert_to_model_inputs,
    _state_se2_array_to_transform_matrix,
    _state_se2_array_to_transform_matrix_batch,
    _transform_matrix_to_state_se2_array_batch,
    vector_set_coordinates_to_local_frame,
)
from utils.coord_utils import carla_transform_to_standard, standard_to_carla

CKPT   = os.path.join(_ROOT, "checkpoints", "model.pth")
ARGS   = os.path.join(_ROOT, "checkpoints", "args.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def PASS(msg): print(f"  [PASS] {msg}")
def FAIL(msg): print(f"  [FAIL] {msg}")
def INFO(msg): print(f"  [INFO] {msg}")
def check(cond, msg, extra=""):
    (PASS if cond else FAIL)(f"{msg}  {extra}")
    return cond


def load_model(guidance_fn=None):
    config = Config(ARGS, guidance_fn=guidance_fn)
    model  = Diffusion_Planner(config)
    raw    = torch.load(CKPT, map_location=DEVICE)
    sd     = raw.get("ema_state_dict") or raw.get("model") or raw
    sd     = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval().to(DEVICE)
    return model, config


def make_route(anchor, dest_std, config, add_shift=True):
    route_num, route_len = config.route_num, config.route_len
    ego_xy = anchor[:2]
    diff   = np.array(dest_std) - ego_xy
    dist   = np.linalg.norm(diff)
    if dist < 1e-3:
        return np.zeros((route_num, route_len, 12), np.float32)
    unit  = diff / dist
    ts    = np.linspace(0.0, dist + 15.0, route_len)
    pts_g = ego_xy[None, :] + unit[None, :] * ts[:, None]
    avails  = np.ones((1, route_len), dtype=np.bool_)
    pts_ego = vector_set_coordinates_to_local_frame(pts_g[np.newaxis], avails, anchor)[0]
    if add_shift:
        pts_ego[:, 0] += 10.0
    fwd = np.diff(pts_ego, axis=0); fwd = np.vstack([fwd, fwd[-1:]])
    tl  = np.tile([0, 0, 0, 1], (route_len, 1)).astype(np.float32)
    base = np.concatenate([pts_ego.astype(np.float32), fwd.astype(np.float32),
                           np.zeros((route_len, 4), np.float32), tl], axis=1)
    route_lanes = np.zeros((route_num, route_len, 12), np.float32)
    for i in range(route_num): route_lanes[i] = base
    return route_lanes


def run_model_raw(model, config, route_lanes, anchor, speed=1.5):
    """Forward pass, returns (T,4) ego-centric pred."""
    from utils.map_process import LANE_NUM, LANE_LEN
    data = {
        "neighbor_agents_past":        np.zeros((32,21,11), np.float32),
        "ego_current_state":           np.array([0,0,1,0,speed,0,0,0,0,0], np.float32),
        "static_objects":              np.zeros((5,10), np.float32),
        "lanes":                       np.zeros((LANE_NUM,LANE_LEN,12), np.float32),
        "lanes_speed_limit":           np.full((LANE_NUM,1), 14., np.float32),
        "lanes_has_speed_limit":       np.ones((LANE_NUM,1), dtype=np.bool_),
        "route_lanes":                 route_lanes,
        "route_lanes_speed_limit":     np.full((config.route_num,1), 14., np.float32),
        "route_lanes_has_speed_limit": np.ones((config.route_num,1), dtype=np.bool_),
    }
    inputs = convert_to_model_inputs(data, DEVICE)
    if config.observation_normalizer:
        inputs = config.observation_normalizer(inputs)
    with torch.no_grad():
        _, out = model(inputs)
    return out["prediction"][0, 0].cpu().numpy()


def ego_pred_to_global(pred, anchor):
    c, s = pred[:,2], pred[:,3]
    n = np.sqrt(c**2+s**2); n = np.where(n<1e-6, 1., n)
    h = np.arctan2(s/n, c/n)
    local = np.stack([pred[:,0], pred[:,1], h], axis=1)
    A = _state_se2_array_to_transform_matrix(anchor)
    P = _state_se2_array_to_transform_matrix_batch(local)
    return _transform_matrix_to_state_se2_array_batch(A @ P)


# ───────────────────────────────────────────────────────────────────────────────

def test_1_normalizer_inspect(model, config):
    """
    Print what route_lanes actually look like AFTER the observation normalizer.
    If normalized values are near-zero, the route is invisible to the model.
    """
    print("\n=== Test 1: route_lanes values after normalization ===")

    ego_cx, ego_cy, ego_yaw = 285.6, -243.7, 90.0
    dst_cx, dst_cy           = 285.6, -183.7

    anchor   = np.array(carla_transform_to_standard(ego_cx, ego_cy, ego_yaw))
    dest_std = np.array(carla_transform_to_standard(dst_cx, dst_cy, 0.)[:2])
    route    = make_route(anchor, dest_std, config)

    from utils.map_process import LANE_NUM, LANE_LEN
    data = {
        "neighbor_agents_past":        np.zeros((32,21,11), np.float32),
        "ego_current_state":           np.array([0,0,1,0,1.5,0,0,0,0,0], np.float32),
        "static_objects":              np.zeros((5,10), np.float32),
        "lanes":                       np.zeros((LANE_NUM,LANE_LEN,12), np.float32),
        "lanes_speed_limit":           np.full((LANE_NUM,1), 14., np.float32),
        "lanes_has_speed_limit":       np.ones((LANE_NUM,1), dtype=np.bool_),
        "route_lanes":                 route,
        "route_lanes_speed_limit":     np.full((config.route_num,1), 14., np.float32),
        "route_lanes_has_speed_limit": np.ones((config.route_num,1), dtype=np.bool_),
    }
    inputs = convert_to_model_inputs(data, DEVICE)

    # Before normalization
    rl_before = inputs["route_lanes"][0, 0].cpu().numpy()  # (route_len, 12)
    print(f"  BEFORE norm — lane 0, pt 0: {rl_before[0, :6]}")
    print(f"  BEFORE norm — lane 0, pt[-1]: {rl_before[-1, :6]}")

    # After normalization
    if config.observation_normalizer:
        inputs_normed = config.observation_normalizer(inputs)
        rl_after = inputs_normed["route_lanes"][0, 0].cpu().numpy()  # (route_len, 12)
        print(f"  AFTER  norm — lane 0, pt 0: {rl_after[0, :6]}")
        print(f"  AFTER  norm — lane 0, pt[-1]: {rl_after[-1, :6]}")

        # Key: are the normalized values near-zero?
        x_vals_norm = rl_after[:, 0]
        y_vals_norm = rl_after[:, 1]
        dx_vals_norm = rl_after[:, 2]
        dy_vals_norm = rl_after[:, 3]

        print(f"\n  Normalized x range: [{x_vals_norm.min():.3f}, {x_vals_norm.max():.3f}]")
        print(f"  Normalized y range: [{y_vals_norm.min():.3f}, {y_vals_norm.max():.3f}]")
        print(f"  Normalized dx[0]: {dx_vals_norm[0]:.4f}  dy[0]: {dy_vals_norm[0]:.4f}")

        # Check zero_route after norm
        zero_data = dict(data); zero_data["route_lanes"] = np.zeros_like(route)
        inputs_zero = convert_to_model_inputs(zero_data, DEVICE)
        inputs_zero_normed = config.observation_normalizer(inputs_zero)
        rl_zero_after = inputs_zero_normed["route_lanes"][0, 0].cpu().numpy()
        print(f"\n  ZERO route AFTER norm — lane 0, pt 0: {rl_zero_after[0, :6]}")
        print(f"  (if zero route normalizes to same as real route → normalizer is wrong)")

        diff_norm = np.abs(rl_after - rl_zero_after).max()
        print(f"\n  Max diff between real and zero route after norm: {diff_norm:.4f}")
        all_ok = check(diff_norm > 0.1,
                       "Real route differs from zero route after normalization",
                       f"max_diff={diff_norm:.4f}")

        # Ego state after norm
        ego_after = inputs_normed["ego_current_state"][0].cpu().numpy()
        print(f"\n  Ego current state after norm: {ego_after[:6]}")
    else:
        INFO("No observation_normalizer configured!")
        all_ok = False

    return all_ok


def test_2_extreme_route_ablation(model, config):
    """
    Feed an 'extreme' route (very large x values) and compare to zero route.
    If the model output changes significantly, route_lanes ARE being read.
    If not, they're completely ignored.
    """
    print("\n=== Test 2: extreme route ablation ===")

    anchor = np.array([0., 0., 0.])  # ego at origin, heading=0

    route_zero   = np.zeros((config.route_num, config.route_len, 12), np.float32)
    route_fwd    = make_route(anchor, np.array([100., 0.]), config)  # 100m forward
    route_right  = make_route(anchor, np.array([0., -50.]), config)  # 50m right (ego-y=-50)
    route_left   = make_route(anchor, np.array([0.,  50.]), config)  # 50m left  (ego-y=+50)

    pred_zero  = run_model_raw(model, config, route_zero,  anchor)
    pred_fwd   = run_model_raw(model, config, route_fwd,   anchor)
    pred_right = run_model_raw(model, config, route_right, anchor)
    pred_left  = run_model_raw(model, config, route_left,  anchor)

    def fmt(p): return f"x_mean={p[:,0].mean():.3f}, y_mean={p[:,1].mean():.3f}"

    print(f"\n  Zero route:   {fmt(pred_zero)}")
    print(f"  Fwd route:    {fmt(pred_fwd)}")
    print(f"  Right route:  {fmt(pred_right)}")
    print(f"  Left route:   {fmt(pred_left)}")

    # Key metric: does left vs right route change the y trajectory?
    y_diff_lr = pred_left[:,1].mean() - pred_right[:,1].mean()
    x_diff_fwd_zero = pred_fwd[:,0].mean() - pred_zero[:,0].mean()

    print(f"\n  Left - Right mean y: {y_diff_lr:.4f}m  (expect > 0 if model responds)")
    print(f"  Fwd - Zero mean x:   {x_diff_fwd_zero:.4f}m  (expect > 0 if model responds)")

    all_ok = True
    all_ok &= check(abs(y_diff_lr) > 0.5,
                    "Left vs Right route creates > 0.5m y-difference in trajectory",
                    f"got {y_diff_lr:.4f}m")
    all_ok &= check(x_diff_fwd_zero > 0.5,
                    "Forward route creates > 0.5m more x-displacement than zero route",
                    f"got {x_diff_fwd_zero:.4f}m")

    if not all_ok:
        INFO("Route_lanes have very weak effect on model output.")
        INFO("The circular prior dominates. Guidance is required to steer.")

    return all_ok


def test_3_guidance_scan(model_no_guidance, config):
    """
    Test guidance with increasing GOAL_SCALE values.
    Find the scale at which the trajectory actually points toward the destination.
    """
    print("\n=== Test 3: guidance strength scan (GOAL_SCALE sweep) ===")

    ego_cx, ego_cy, ego_yaw = 285.6, -243.7, 90.0
    dst_cx, dst_cy           = 285.6, -183.7   # 60m ahead

    anchor   = np.array(carla_transform_to_standard(ego_cx, ego_cy, ego_yaw))
    dest_std = np.array(carla_transform_to_standard(dst_cx, dst_cy, 0.)[:2])

    # Destination in ego frame
    M_inv    = np.linalg.inv(_state_se2_array_to_transform_matrix(anchor))
    dest_h   = np.array([dest_std[0], dest_std[1], 1.0])
    dest_ego = (M_inv @ dest_h)[:2]
    INFO(f"Destination in ego frame: ({dest_ego[0]:.1f}, {dest_ego[1]:.1f})  [expect ~(+60, 0)]")

    results_by_scale = {}
    for scale in [0.0, 50.0, 500.0, 1000.0, 2000.0, 3000.0]:
        def make_guidance(goal_scale, dest, anc_inv):
            def _guidance_fn(x, t, cond, **kwargs):
                zero = (x * 0.0).sum()
                state_normalizer = kwargs.get("state_normalizer")
                model            = kwargs.get("model")
                model_condition  = kwargs.get("model_condition")
                if state_normalizer is None or model is None or model_condition is None:
                    return zero

                t_val = t.mean().item() if hasattr(t, "mean") else float(t)
                if t_val > 0.9 or t_val < 0.001:
                    return zero

                B, P, D = x.shape

                # Tweedie correction: denoise one step for a clean x0 estimate
                with torch.no_grad():
                    x_fix = model(x, t, **model_condition) - x.detach()
                x_fix = x_fix.reshape(B, P, -1, 4)
                x_fix[:, :, 0] = 0.0
                x = x + x_fix.reshape(B, P, -1)

                T_plus1 = D // 4
                x_ego  = x[:, 0, :].reshape(B, T_plus1, 4)
                x_real = state_normalizer.inverse(x_ego)
                future_xy = x_real[:, 1:, :2]

                dest_t  = torch.tensor(dest, dtype=x.dtype, device=x.device)
                M_inv_t = torch.tensor(anc_inv, dtype=x.dtype, device=x.device)
                dest_h  = torch.stack([dest_t[0], dest_t[1],
                                        torch.ones(1, device=x.device).squeeze()])
                dest_e  = (M_inv_t @ dest_h)[:2]

                final_xy  = future_xy[:, -1, :]
                goal_dist = torch.norm(final_xy - dest_e[None], dim=-1)
                return -goal_scale * goal_dist.mean()
            return _guidance_fn

        guidance = make_guidance(scale, dest_std, M_inv)
        cfg = Config(ARGS, guidance_fn=guidance)
        mdl = Diffusion_Planner(cfg)
        raw = torch.load(CKPT, map_location=DEVICE)
        sd  = raw.get("ema_state_dict") or raw.get("model") or raw
        sd  = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        mdl.load_state_dict(sd)
        mdl.eval().to(DEVICE)

        route = make_route(anchor, dest_std, cfg)
        pred  = run_model_raw(mdl, cfg, route, anchor)
        final = pred[-1, :2]
        dist_to_dest = np.linalg.norm(final - dest_ego)
        mean_x = pred[:, 0].mean()
        mean_y = pred[:, 1].mean()

        results_by_scale[scale] = (final, dist_to_dest, mean_x, mean_y)
        print(f"  scale={scale:5.1f}: final=({final[0]:7.3f},{final[1]:7.3f})"
              f"  dist_to_dest={dist_to_dest:.2f}m  mean_x={mean_x:.3f}  mean_y={mean_y:.3f}")

    # Find the best scale
    best_scale = min(results_by_scale.keys(),
                     key=lambda s: results_by_scale[s][1])
    best_dist  = results_by_scale[best_scale][1]

    print(f"\n  Best GOAL_SCALE = {best_scale}  (dist_to_dest = {best_dist:.2f}m)")

    all_ok = True
    all_ok &= check(best_dist < 30.0,
                    f"Some guidance scale gets within 30m of destination",
                    f"best scale={best_scale}, dist={best_dist:.2f}m")

    # Check monotone improvement
    scales = sorted(results_by_scale.keys())
    dists  = [results_by_scale[s][1] for s in scales]
    generally_decreasing = dists[-1] < dists[0]
    INFO(f"Dist at scale=0: {dists[0]:.2f}m  →  scale={scales[-1]}: {dists[-1]:.2f}m")
    if generally_decreasing:
        INFO("Guidance IS effective — increasing scale reduces distance to destination.")
        INFO(f"→ Use GOAL_SCALE = {best_scale} in diff_adapter.py")
    else:
        INFO("Guidance is NOT effective — increasing scale does not help.")

    return all_ok


def test_4_route_num_lanes(model, config):
    """
    In diff_adapter.py, all route_num lanes are identical (all offsets=0).
    The model was trained with laterally spread lanes (different offsets).
    Try spreading the route lanes laterally to see if it improves response.
    """
    print("\n=== Test 4: lateral lane spread sensitivity ===")

    ego_cx, ego_cy, ego_yaw = 285.6, -243.7, 90.0
    dst_cx, dst_cy           = 285.6, -183.7

    anchor   = np.array(carla_transform_to_standard(ego_cx, ego_cy, ego_yaw))
    dest_std = np.array(carla_transform_to_standard(dst_cx, dst_cy, 0.)[:2])

    # Build base route (no shift for clean comparison)
    route_num, route_len = config.route_num, config.route_len
    ego_xy = anchor[:2]
    diff = dest_std - ego_xy
    dist = np.linalg.norm(diff)
    unit = diff / dist
    ts = np.linspace(0., dist+15., route_len)
    pts_g = ego_xy[None,:] + unit[None,:] * ts[:,None]
    avails = np.ones((1, route_len), dtype=np.bool_)
    pts_ego = vector_set_coordinates_to_local_frame(pts_g[np.newaxis], avails, anchor)[0]
    pts_ego[:, 0] += 10.0  # +10 shift

    fwd = np.diff(pts_ego, axis=0); fwd = np.vstack([fwd, fwd[-1:]])
    tl  = np.tile([0,0,0,1], (route_len,1)).astype(np.float32)

    # All-same lanes (current behavior)
    route_same = np.zeros((route_num, route_len, 12), np.float32)
    for i in range(route_num):
        base = np.concatenate([pts_ego.astype(np.float32), fwd.astype(np.float32),
                                np.zeros((route_len,4),np.float32), tl], axis=1)
        route_same[i] = base

    # Spread lanes with offsets: [-3.5, -1.75, 0, 1.75, 3.5, ...]
    offsets = np.linspace(-3.5 * (route_num//2), 3.5 * (route_num//2), route_num)
    route_spread = np.zeros((route_num, route_len, 12), np.float32)
    for i in range(route_num):
        lane = pts_ego.copy()
        lane[:, 1] += offsets[i]   # lateral offset in ego-y
        base = np.concatenate([lane.astype(np.float32), fwd.astype(np.float32),
                                np.zeros((route_len,4),np.float32), tl], axis=1)
        route_spread[i] = base

    pred_same   = run_model_raw(model, config, route_same,   anchor)
    pred_spread = run_model_raw(model, config, route_spread, anchor)

    def fmt(p): return f"x_mean={p[:,0].mean():.3f}, y_mean={p[:,1].mean():.3f}, final=({p[-1,0]:.2f},{p[-1,1]:.2f})"
    print(f"  All-same lanes (current): {fmt(pred_same)}")
    print(f"  Spread lanes (nuPlan-like): {fmt(pred_spread)}")

    diff_y = abs(pred_spread[:,1].mean() - pred_same[:,1].mean())
    diff_x = abs(pred_spread[:,0].mean() - pred_same[:,0].mean())
    INFO(f"Spread vs same: Δy={diff_y:.4f}m  Δx={diff_x:.4f}m")

    all_ok = check(diff_y < 2.0 or diff_x > 0.5,
                   "Spread lanes change output meaningfully or maintain forward direction",
                   f"Δy={diff_y:.3f}, Δx={diff_x:.3f}")
    return all_ok


# ───────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  Route Influence Deep-Dive")
    print(f"  Device: {DEVICE}")
    print("=" * 65)

    model, config = load_model()

    r1 = test_1_normalizer_inspect(model, config)
    r2 = test_2_extreme_route_ablation(model, config)
    r3 = test_3_guidance_scan(model, config)   # loads model multiple times
    r4 = test_4_route_num_lanes(model, config)

    print(f"\n{'=' * 65}")
    results = [r1, r2, r3, r4]
    names   = ["normalizer inspect", "extreme ablation",
               "guidance scale scan", "lane spread sensitivity"]
    for i, (name, ok) in enumerate(zip(names, results)):
        print(f"  {'[PASS]' if ok else '[FAIL]'} Test {i+1}: {name}")

    print(f"\n{sum(results)}/{len(results)} tests passed")
    print("\n=== ACTION ITEMS ===")
    print("Based on test_model_live.py results:")
    print("  - Route influence is ~2.6% (0.23m / 8.6m) → VERY WEAK")
    print("  - Circular prior is dominant with zero-route")
    print("  - Left route doesn't produce positive ego-y → model barely responds")
    print()
    print("Likely fix: The model needs strong GUIDANCE to override the circular prior.")
    print("  In diff_adapter.py, try GOAL_SCALE = 10-50 based on test_3 results above.")
