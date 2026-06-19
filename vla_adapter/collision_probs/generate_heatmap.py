"""
BEV latency-aware collision-risk heatmap (LICOM), following LAVQA Sec. III-A/B.

For every BEV cell -- treated as a *candidate ego position* -- we compute the
probability that the ego footprint placed there intersects any dynamic obstacle
whose future pose is a (multi-modal) Gaussian mixture. This is a real
probability in [0, 1]: P(E n O != empty) from Definition 3, NOT a raw density.

Why not the obstacle pdf directly? scipy's multivariate_normal.pdf returns a
density whose peak scales like 1/sqrt(|Sigma|): a confident (tight) obstacle
spikes, a diffuse one vanishes, and the value has units of 1/m^2 -- so any fixed
gain + clamp breaks as the covariance changes. The collision *probability* is
the Gaussian *mass* inside the ego(+)obstacle Minkowski region, which is bounded
in [0, 1] and needs no magic multiplier.

Coordinate convention matches agent_adapter.ObstacleMap:
    grid axis 0 <-> world x,  grid axis 1 <-> world y,  resolution 0.25 m.

GMM data model (one entry per obstacle):
    {
      'half_length': float,   # obstacle footprint half-extents (m)
      'half_width':  float,
      'modes': [ {'weight': Psi_j, 'mu': (x, y), 'cov': 2x2}, ... ],  # sum Psi_j = 1
    }
"""
import numpy as np
from scipy.special import ndtr as _ndtr   # raw CDF with no input-validation overhead

from config import (
    EGO_HALF_LENGTH, EGO_HALF_WIDTH,
    DEF_OBS_HALF_LENGTH, DEF_OBS_HALF_WIDTH,
)

## Personal Notes
### main information to be conveyed to the VLA: If the ego vehicle were at position (x, y), what's the probability it collides with a nearby obstacle?"

### Instead of using a PDF, we use a Minkowski sum box, the combined footprint of the ego + obstacle. We check if there is a collision between
## these bounding boxes via separating axis theorem.

### How do we do this?
## Main logic is _mode_collision_prob: 



def make_bev_grid(min_x, min_y, shape, resolution=0.25):
    """Cell-centre world coords, aligned with an ObstacleMap grid.

    Returns (gx, gy), each of `shape`, axis 0 = x, axis 1 = y.
    """
    nx, ny = shape
    xs = min_x + (np.arange(nx) + 0.5) * resolution
    ys = min_y + (np.arange(ny) + 0.5) * resolution
    return np.meshgrid(xs, ys, indexing='ij')


def grid_from_obstacle_map(obs_map, resolution=0.25):
    """Build the BEV grid that lines up cell-for-cell with an ObstacleMap."""
    return make_bev_grid(obs_map.min_x, obs_map.min_y, obs_map.obs.shape, resolution)

### pprobability that obstacles center lands in the footprint
def _mode_collision_prob(gx, gy, mu, cov, half_a, half_b, travel_dir=None):
    """Gaussian mass of one GMM mode inside the ego(+)obstacle Minkowski box.

    When travel_dir is provided, the Minkowski box is oriented along that
    direction (half_a longitudinal, half_b lateral), computed via direct
    projection: var_lon = t @ cov @ t. This works correctly for both lateral
    crossings (pedestrians) and head-on vehicles without any eigenvalue tricks.

    When travel_dir is None, falls back to the eigenvalue-based approximation.
    """
    ### For every grid cell (gx, gy), if the ego vehicle is centered here what is the probabilitity that the Gaussian
    ## distributed obstacle intersects it?  

    ## the collision probability = Gaussian mass inside the Minkowski box. If the obstacle center lands inside it, collision
    ## is guaranteed regardless of relative orientations
    ## Make it axis aligned:
    ## P(μ in R) = P(lon ∈ [p_lon − a, p_lon + a]) × P(lat ∈ [p_lat − b, p_lat + b])
    ## = Φ((a + p_lon)/σ_lon) − Φ((−a + p_lon)/σ_lon)] × [same for lat]

    cov = np.asarray(cov, dtype=float)
    ## where is the grid cell relative to where the obstacle expects to be?
    offset = np.stack([gx - mu[0], gy - mu[1]], axis=-1)  # (H, W, 2), every grid cell gets a 2D vector pointing from the obstacle's mean μ to that cell

    if travel_dir is not None:
        t = np.asarray(travel_dir, dtype=float)
        ## build unit vector along travel direction
        t = t / (np.linalg.norm(t) + 1e-9)
        ### normal to travel unit vector. negative recirprocal slope
        perp = np.array([-t[1], t[0]])
        ## directional standard deviations using rayleigh quotient (gives the variance of the distribution when you project it onto direction t)
        std_lon = float(np.sqrt(max(float(t @ cov @ t), 1e-6)))
        std_lat = float(np.sqrt(max(float(perp @ cov @ perp), 1e-6)))
        ### how far is the cell from μ in the longitudinal and lateral projections
        p_lon = offset @ t     # (H, W)
        p_lat = offset @ perp  # (H, W)
        ## We want P(obstacle centre lands within half_a of the grid cell, along the longitudinal axis).
        ## we want the probability mass between p_lon - half_a and p_lon + half_a in the distribution centred at 0 (centered at μ)
        ## will a sample from N(0, σ²) land in the interval [-half_a - p_lon, half_a - p_lon]
        prob_lon = np.clip(_ndtr((half_a + p_lon) / std_lon) - _ndtr((-half_a + p_lon) / std_lon), 0.0, 1.0)
        prob_lat = np.clip(_ndtr((half_b + p_lat) / std_lat) - _ndtr((-half_b + p_lat) / std_lat), 0.0, 1.0)
        return prob_lon * prob_lat
    else:
        # Legacy fallback: eigenvector-based box (assumes lateral crossing).
        print("No travel direction provided")

### Full GMM for one obstacle, weighted sum over modes
def obstacle_collision_prob(gx, gy, obstacle,
                            ego_half_len=EGO_HALF_LENGTH,
                            ego_half_wid=EGO_HALF_WIDTH):
    """Marginal collision probability for one obstacle = sum_j Psi_j P_box(mode j)."""
    obs_hl = obstacle.get('half_length', DEF_OBS_HALF_LENGTH)
    obs_hw = obstacle.get('half_width', DEF_OBS_HALF_WIDTH)
    half_a = ego_half_len + obs_hl   # combined longitudinal extent (Minkowski)
    half_b = ego_half_wid + obs_hw   # combined lateral extent
    p = np.zeros_like(gx)
    ### Weighted mean of collision mode probabilities (bayes law, each mode is mutually exclusive)
    for mode in obstacle['modes']:
        p = p + mode['weight'] * _mode_collision_prob(
            gx, gy, mode['mu'], mode['cov'], half_a, half_b,
            travel_dir=mode.get('travel_dir'))
    return np.clip(p, 0.0, 1.0)

## takes the elementwise MAX, giving you the swept hazard zone across all considered future times
def licom_risk(gx, gy, obstacle_snapshots, **kw):
    """Latency-aware collision map over a BEV grid.

    obstacle_snapshots: either a single snapshot (list of obstacle dicts) or a
        list of snapshots, one per response-latency / horizon sample. Across a
        snapshot, obstacles combine as an independent union 1 - prod(1 - P_i);
        across snapshots we take the elementwise MAX -- i.e. the swept hazard
        zone that "expands and shifts" as obstacles are propagated forward,
        which is exactly the latency-induced risk LAVQA visualises.

    Returns a float risk grid in [0, 1], same shape as gx.
    """
    if obstacle_snapshots and isinstance(obstacle_snapshots[0], dict):
        obstacle_snapshots = [obstacle_snapshots]

    risk = np.zeros_like(gx)
    for snapshot in obstacle_snapshots:
        survive = np.ones_like(gx)
        for obstacle in snapshot:
            survive *= (1.0 - obstacle_collision_prob(gx, gy, obstacle, **kw))
        risk = np.maximum(risk, 1.0 - survive)
    return risk

### NOT USED IN DATA COLLECTION (started to think about inference time)
def gmm_from_kf(state_mean, state_cov, horizon, q=0.1,
                half_length=DEF_OBS_HALF_LENGTH, half_width=DEF_OBS_HALF_WIDTH):
    """Inference-time J=1 GMM: propagate an agent_adapter KF Gaussian forward.

    Reuses the constant-velocity model already in agent_adapter
    (state = [x, y, vx, vy], transition cov q*I): mu' = F mu,
    Sigma' = F Sigma F^T + Q*horizon. Use this when you don't have the
    privileged map-aware predictor (e.g. live, from dyn_obs_clusters).
    """
    F = np.array([[1, 0, horizon, 0],
                  [0, 1, 0, horizon],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=float)
    m = F @ np.asarray(state_mean, dtype=float)
    P = F @ np.asarray(state_cov, dtype=float) @ F.T + q * horizon * np.eye(4)
    return {'half_length': half_length, 'half_width': half_width,
            'modes': [{'weight': 1.0, 'mu': m[:2], 'cov': P[:2, :2]}]}

### heatmap colors
def colorize(risk, colormap=False):
    """risk float[0,1] -> uint8 image for stacking onto the VLA input.

    NOTE: like ObstacleMap, axis 0 = x grows downward; flip/transpose to taste
    when saving for human inspection (the model only needs consistency).
    """
    g = np.uint8(np.clip(risk, 0.0, 1.0) * 255)
    if not colormap:
        return g
    import cv2
    return cv2.applyColorMap(g, cv2.COLORMAP_JET)


if __name__ == "__main__":
    # Smoke test: one obstacle, two modes (straight vs. left), 0.5 s latency.
    gx, gy = make_bev_grid(min_x=-20.0, min_y=-20.0, shape=(160, 160))
    obstacle = {
        'half_length': 2.3, 'half_width': 1.0,
        'modes': [
            {'weight': 0.7, 'mu': (5.0, 0.0), 'cov': np.array([[2.0, 0.0], [0.0, 0.4]])},
            {'weight': 0.3, 'mu': (3.0, 4.0), 'cov': np.array([[1.0, 0.5], [0.5, 1.0]])},
        ],
    }
    risk = licom_risk(gx, gy, [obstacle])
    print("risk range:", float(risk.min()), float(risk.max()), "shape:", risk.shape)
