"""
Build the swept (max-over-latency-window) collision-risk heatmap in an
ego-centric forward-up BEV frame.

Obstacle-type agnostic: each entry is a generic GMM "obstacle" dict and may be a
pedestrian (PedestrianCrossingParking) or a vehicle (opposite_vehicle_parking).
The math only reads the dict fields below; type-specific behaviour is carried in
the field values (e.g. p_cross=0.5 vs 1/3, cone_rate=0.18 vs 0.05).

Pipeline (all decisions locked with the user):
  - GMM source: per-obstacle candidate futures emitted by the scenario, in WORLD
    frame: {edge_point, cross_dir (unit), cross_distance, speed, p_cross, extent}.
  - Modes (per obstacle): CROSS (the obstacle sweeps the lane) with weight p_cross,
    and STAY (waits at the lane edge) with weight 1 - p_cross. Cross/not-cross is
    the only multimodal axis we model.
  - Crossing motion: a WIDENING REACHABILITY CONE. We sample the crossing over a
    time window t in [0, T]; at each sample the mean advances along cross_dir and
    the LATERAL covariance grows with downrange distance (the fan). Longitudinal
    spread is supplied by the time-sweep itself (we take the per-cell max over
    samples), so longitudinal positional variance stays footprint-small -- no
    double counting. This is sup over the latency window of LAVQA's LICOM(g; tau).
  - Collision probability: the box / Minkowski Gaussian-mass integral in
    generate_heatmap (a convex-hull approximation of P(E n O != empty)).
  - Multiple obstacles: independent union 1 - prod(1 - P_i).
  - Frame: ego-centric, x = forward, y = left (matches transfuser inverse_conversion_2d).

Coordinates passed to the renderer are in EGO frame, so the ego footprint
(EGO_HALF_LENGTH/WIDTH, added inside generate_heatmap.obstacle_collision_prob)
sits at the cell under test -- exactly the ego region E in Definition 3.
"""
import numpy as np
import cv2

from generate_heatmap import licom_risk, colorize
from config import (
    HORIZON_SCALE, TIME_HORIZON, N_TIME,
    SIGMA_LON, SIGMA_LAT0, CONE_RATE, STAY_SIGMA,
    EGO_HALF_LENGTH, EGO_HALF_WIDTH, CORRIDOR_SAFE_FLOOR,
    VEHICLE_WHEELBASE, MAX_STEER_DEG, CONE_REACH, CONE_HALF_ANGLE_DEG,
    STATIC_EGO_INFLATE_M, STATIC_SPEED_EPS,
)


# ---------------------------------------------------------------------------
# Vibrant green → red colormap (replaces JET).
# risk=0.0 → vibrant green RGB (40, 210, 40)
# risk=0.5 → amber         RGB (220, 200, 0)
# risk=1.0 → soft red      RGB (220, 30,  30)
# ---------------------------------------------------------------------------

# Risk at/below this is "safe" and renders flat green; everything above it is
# the warm warning ramp (yellow -> orange -> red). Keep it just above the
# corridor's CORRIDOR_SAFE_FLOOR (0.10) so the safe corridor stays green.
_GREEN_THR = 0.15


def _build_soft_rg_lut():
    """256-entry BGR lookup table: a flat-green SAFE band below _GREEN_THR, then
    a single warm warning ramp yellow -> orange -> red above it.

    Only one hard cut, green -> yellow, at the threshold. The warm ramp is one
    hue family (red pinned high, green channel falling) so it gradates smoothly
    yellow->orange->red without the muddy cross-hue blending we had before.
        risk <= _GREEN_THR  -> green   (safe)
        risk  > _GREEN_THR  -> yellow -> orange -> red, by intensity
    """
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t <= _GREEN_THR:                # SAFE -> flat vivid green
            r, g, b = 0, 200, 0
        else:
            s = (t - _GREEN_THR) / (1.0 - _GREEN_THR)   # 0..1 across the warm ramp
            if s < 0.5:                    # yellow -> orange
                u = s / 0.5
                r, g, b = 255, int(235 - u * (235 - 140)), 0
            else:                          # orange -> red
                u = (s - 0.5) / 0.5
                r, g, b = int(255 - u * (255 - 205)), int(140 * (1.0 - u)), 0
        lut[i, 0] = (b, g, r)              # OpenCV uses BGR
    return lut

_SOFT_RG_LUT = _build_soft_rg_lut()


def colorize_soft(risk):
    """risk float[0,1] ndarray -> uint8 BGR image with soft green/red palette."""
    gray = np.uint8(np.clip(risk, 0.0, 1.0) * 255)
    # cv2.LUT needs a 3-channel input to produce a 3-channel output.
    gray3 = np.stack([gray, gray, gray], axis=-1).reshape(-1, 1, 3)
    return cv2.LUT(gray3, _SOFT_RG_LUT).reshape(gray.shape[0], gray.shape[1], 3)


# ---------------------------------------------------------------------------
# Ego-frame transforms (x = forward, y = left), matching transfuser_utils.
# ---------------------------------------------------------------------------

def world_to_ego(pt, ego_xy, ego_yaw):
    d = np.asarray(pt, dtype=float) - np.asarray(ego_xy, dtype=float)
    c, s = np.cos(ego_yaw), np.sin(ego_yaw)
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])


def rot_to_ego(vec, ego_yaw):
    c, s = np.cos(ego_yaw), np.sin(ego_yaw)
    v = np.asarray(vec, dtype=float)
    return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1]])


# ---------------------------------------------------------------------------
# GMM construction
# ---------------------------------------------------------------------------

def _cone_cov(direction, dist, sigma_lon, sigma_lat0, cone_rate):
    """Positional covariance for the cone at downrange distance `dist`.

    Longitudinal (along `direction`) variance stays small -- the time-sweep
    supplies that extent. Lateral (perpendicular) std grows: sigma_lat0 +
    cone_rate * dist, which is the fan opening up downrange.
    """
    direction = np.asarray(direction, dtype=float)
    direction = direction / (np.linalg.norm(direction) + 1e-9)
    perp = np.array([-direction[1], direction[0]])
    sig_lat = sigma_lat0 + cone_rate * dist
    return (sigma_lon ** 2) * np.outer(direction, direction) \
        + (sig_lat ** 2) * np.outer(perp, perp)


def build_crossing_snapshots(obstacles, ego_xy, ego_yaw,
                             n_time=N_TIME,
                             horizon_scale=HORIZON_SCALE,
                             time_horizon=TIME_HORIZON,
                             sigma_lon=SIGMA_LON,
                             sigma_lat0=SIGMA_LAT0,
                             cone_rate=CONE_RATE,
                             stay_sigma=STAY_SIGMA):
    """Time-sampled GMM snapshots (ego frame) for licom_risk.

    Projects each obstacle forward at its stated speed over time_horizon seconds.
    dist = speed × t, capped at cross_distance × horizon_scale so the sweep
    never extends beyond the scenario's intended range.

    This means:
      - fast obstacles reach far ahead quickly → wide, far-reaching corridor
      - slow/stopped obstacles stay near current position → short/no corridor
      - p_cross weights corridor (CROSS mode) vs stationary blob (STAY mode)
    """
    geom = []
    for obs in obstacles:
        edge = world_to_ego(obs['edge_point'], ego_xy, ego_yaw)
        d = rot_to_ego(obs['cross_dir'], ego_yaw)
        d = d / (np.linalg.norm(d) + 1e-9)
        geom.append((obs, edge, d))

    times = np.linspace(0.0, time_horizon, n_time)
    snapshots = []
    for t in times:
        snap = []
        for obs, edge, d in geom:
            speed    = float(obs.get('speed', 1.3))
            max_dist = float(obs['cross_distance']) * horizon_scale
            dist     = min(speed * t, max_dist)
            # Per-obstacle cone_rate override (e.g. vehicles 0.05, pedestrians 0.18).
            obs_cone_rate = float(obs.get('cone_rate', cone_rate))
            # travel_dir lets _mode_collision_prob orient the Minkowski box correctly
            # along the obstacle's actual direction of motion, not by eigenvalue ordering.
            travel_dir = d.tolist()
            modes = [{
                'weight': float(obs['p_cross']),
                'mu': edge + dist * d,
                'cov': _cone_cov(d, dist, sigma_lon, sigma_lat0, obs_cone_rate),
                'travel_dir': travel_dir,
            }]
            if obs['p_cross'] < 1.0:
                modes.append({
                    'weight': float(1.0 - obs['p_cross']),
                    'mu': edge,
                    'cov': (stay_sigma ** 2) * np.eye(2),
                    'travel_dir': travel_dir,
                })
            ext = obs.get('extent', [0.3, 0.3])
            snap.append({'half_length': float(ext[0]),
                         'half_width':  float(ext[1]),
                         'modes': modes})
        snapshots.append(snap)
    return snapshots


# ---------------------------------------------------------------------------
# Ego-centric BEV rendering
# ---------------------------------------------------------------------------

def clip_route_to_bumper(route_pts, ego_half_length=EGO_HALF_LENGTH):
    """Return the subset of route_pts starting from the front/rear bumper.

    route_pts are in ego frame ([x_fwd, y_left]).  route[0] is always [0,0]
    (car centre), which makes the corridor wrap around the ego's sides.
    This clips the route so the corridor starts at the bumper instead.

    Direction is inferred from how the route BEGINS, not where it ends -- routes
    can change direction mid-maneuver (e.g. a forward pull-up followed by a
    reverse correction), and the heatmap should start at whichever bumper the
    car is about to move toward right now. We scan forward from route[1] for
    the first point with a non-negligible x offset (skipping near-zero noise
    right at the car centre) and use its sign.
    """
    if not route_pts or len(route_pts) < 2:
        return route_pts
    pts = np.array(route_pts, dtype=float)
    eps = 0.1 * ego_half_length
    nonzero = np.flatnonzero(np.abs(pts[1:, 0]) > eps)
    start_x = pts[1 + nonzero[0], 0] if len(nonzero) else pts[-1, 0]
    going_forward = start_x >= 0
    threshold = ego_half_length if going_forward else -ego_half_length
    if going_forward:
        clipped = [pt for pt in route_pts if pt[0] >= threshold]
    else:
        clipped = [pt for pt in route_pts if pt[0] <= threshold]
    return clipped if clipped else route_pts


def route_corridor_mask(route_pts, half_width, topdown_half, size):
    """Boolean mask (size×size) True where the pixel lies within half_width metres
    of the route polyline, in the forward-up ego image after to_forward_up_image.

    Coordinate mapping for the (size×size) image:
        x_ego =  topdown_half * (1 - 2*row/size)   # row 0 = max forward
        y_ego =  topdown_half * (2*col/size - 1)   # col 0 = rightmost
    """
    rows = np.arange(size, dtype=float)
    cols = np.arange(size, dtype=float)
    col_g, row_g = np.meshgrid(cols, rows)
    px_x = topdown_half * (1.0 - 2.0 * row_g / size)
    px_y = topdown_half * (2.0 * col_g / size - 1.0)

    pts = np.array(route_pts, dtype=float)   # (N, 2): [x_fwd, y_left]
    min_dist_sq = np.full((size, size), np.inf)
    hw_sq = half_width ** 2

    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        ab = b - a
        ab_len_sq = float(ab @ ab)
        if ab_len_sq < 1e-9:
            min_dist_sq = np.minimum(min_dist_sq,
                                     (px_x - a[0]) ** 2 + (px_y - a[1]) ** 2)
            continue
        t = np.clip(((px_x - a[0]) * ab[0] + (px_y - a[1]) * ab[1]) / ab_len_sq,
                    0.0, 1.0)
        dx = px_x - (a[0] + t * ab[0])
        dy = px_y - (a[1] + t * ab[1])
        min_dist_sq = np.minimum(min_dist_sq, dx * dx + dy * dy)
        if np.all(min_dist_sq <= hw_sq):
            break

    return min_dist_sq <= hw_sq


def reachable_cone_mask(ego_speed, topdown_half, size,
                        half_angle_deg=CONE_HALF_ANGLE_DEG,
                        reach=CONE_REACH, base_half=EGO_HALF_WIDTH):
    """Boolean mask (size×size) of the ego's forward REACHABLE fan -- the cone of
    positions it could drive to, in the same forward-up ego image coords as
    route_corridor_mask. Symmetric in x (a forward + reverse bow-tie) since
    parking reverses (into_spot/recovery/correction).

    `reach` is a FIXED longitudinal distance each way (no speed dependence);
    `ego_speed` is accepted but unused (kept for call-site stability).

    Geometry: a true cone. The lateral half-width opens LINEARLY with downrange
    distance, hw(x) = base_half + tan(half_angle) * |x|, giving straight angled
    sides from a vehicle-wide base at the bumper. (The earlier min-turn-radius
    arc, r_min - sqrt(r_min² - x²), saturated at r_min and clipped past it, so it
    rendered as a box -> flare -> box with square sides rather than a cone.)
    """
    reach = min(reach, topdown_half)   # FIXED distance, both directions; ego_speed unused
    slope = np.tan(np.deg2rad(half_angle_deg))

    rows = np.arange(size, dtype=float)
    cols = np.arange(size, dtype=float)
    col_g, row_g = np.meshgrid(cols, rows)
    px_x = topdown_half * (1.0 - 2.0 * row_g / size)   # longitudinal (row 0 = max fwd)
    px_y = topdown_half * (2.0 * col_g / size - 1.0)   # left/right

    hw = base_half + slope * np.abs(px_x)              # linear fan -> straight cone sides
    return (np.abs(px_x) <= reach) & (np.abs(px_y) <= hw)   # forward + reverse lobes


def render_ego_bev(obstacles, ego_xy, ego_yaw,
                   x_range=(-5.0, 25.0), y_range=(-10.0, 10.0),
                   resolution=0.25, **snapshot_kw):
    """Return (risk, (gx, gy)) on an ego-frame grid (axis0=x fwd, axis1=y left)."""
    xs = np.arange(x_range[0], x_range[1], resolution) + resolution / 2.0
    ys = np.arange(y_range[0], y_range[1], resolution) + resolution / 2.0
    gx, gy = np.meshgrid(xs, ys, indexing='ij')
    if not obstacles:
        return np.zeros_like(gx), (gx, gy)
    snaps = build_crossing_snapshots(obstacles, ego_xy, ego_yaw, **snapshot_kw)
    return licom_risk(gx, gy, snaps), (gx, gy)


def render_world_bev(obstacles, x_range, y_range, resolution=0.25, **snapshot_kw):
    """Danger-zone risk on a WORLD-frame grid (for top-down visualization)."""
    xs = np.arange(x_range[0], x_range[1], resolution) + resolution / 2.0
    ys = np.arange(y_range[0], y_range[1], resolution) + resolution / 2.0
    gx, gy = np.meshgrid(xs, ys, indexing='ij')
    if not obstacles:
        return np.zeros_like(gx), xs, ys
    snaps = build_crossing_snapshots(obstacles, np.zeros(2), 0.0, **snapshot_kw)
    return licom_risk(gx, gy, snaps), xs, ys


def to_forward_up_image(risk):
    """Ego grid -> image with forward UP. Only flips axis 0 (forward).

    Axis 1 is left as-is: negative-y (physical left in CARLA) stays at low
    column indices = left in image, matching the topdown camera where
    col 0 = car's physical left (camera x-axis = car's right = high col index).
    """
    return risk[::-1, :]


def render_ego_bev_image(obstacles, ego_xy, ego_yaw,
                         colormap=True, soft_colors=True, **kw):
    """Ego-centric forward-up heatmap ready to save.

    colormap=True  + soft_colors=True  -> soft green/red BGR image  (default)
    colormap=True  + soft_colors=False -> JET BGR image
    colormap=False                     -> grayscale uint8
    """
    risk, _ = render_ego_bev(obstacles, ego_xy, ego_yaw, **kw)
    fwd = to_forward_up_image(risk)
    if not colormap:
        return colorize(fwd, colormap=False)   # grayscale
    if soft_colors:
        return colorize_soft(fwd)              # soft green/red  ← new default
    return colorize(fwd, colormap=True)        # original JET


def static_obstacle_mask(boxes_ego, size, topdown_half,
                         ego_inflate=STATIC_EGO_INFLATE_M,
                         speed_eps=STATIC_SPEED_EPS):
    """Ego-frame static boxes -> boolean cut-out mask on the size×size forward-up
    grid, in the SAME image convention as reachable_cone_mask (row 0 = max
    forward, col 0 = physical left). True where a parked car sits, so the caller
    can punch a hole in the heatmap there and let it flow around the car.

    boxes_ego: list of {'position':[x_fwd,y_lat,...], 'extent':[ex,ey,...],
    'yaw':rad_rel_ego, 'speed':m/s, 'class':...} -- exactly what save_boxes
    writes (ego frame via inverse_conversion_2d == world_to_ego). Only boxes
    with speed < speed_eps are used (parked cars). Footprints are inflated by
    the ego half-width (Minkowski) so the hole reflects where the ego centre
    could not go; set ego_inflate=0 for the bare car footprint.
    """
    px_per_m = size / (2.0 * topdown_half)

    def to_rc(x_fwd, y_lat):
        # Inverse of reachable_cone_mask's px_x/px_y mapping.
        row = (size / 2.0) * (1.0 - x_fwd / topdown_half)
        col = (size / 2.0) * (y_lat / topdown_half + 1.0)
        return col, row   # cv2 fillPoly wants (x=col, y=row)

    obs = np.zeros((size, size), dtype=np.uint8)
    for b in (boxes_ego or []):
        if float(b.get('speed', 0.0)) >= speed_eps:
            continue
        bx, by = float(b['position'][0]), float(b['position'][1])
        ex, ey = float(b['extent'][0]), float(b['extent'][1])
        yaw = float(b.get('yaw', 0.0))
        c, s = np.cos(yaw), np.sin(yaw)
        pts = []
        for sx, sy in ((+ex, +ey), (+ex, -ey), (-ex, -ey), (-ex, +ey)):
            # box-local corner -> ego frame -> image (col,row)
            pts.append(to_rc(bx + c * sx - s * sy, by + s * sx + c * sy))
        cv2.fillConvexPoly(obs, np.round(pts).astype(np.int32), 255)

    if not obs.any():
        return np.zeros((size, size), bool)

    rad = int(round(ego_inflate * px_per_m))
    if rad > 0:                            # Minkowski-inflate by the ego half-width
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
        obs = cv2.dilate(obs, k)
    return obs > 0


def render_route_risk_grid(obstacles, ego_xy, ego_yaw, route_pts,
                           size, topdown_half, corridor_half_width,
                           resolution=1.0, horizon_scale=HORIZON_SCALE,
                           corridor_floor=CORRIDOR_SAFE_FLOOR,
                           window='corridor', ego_speed=0.0,
                           static_boxes=None):
    """Coarse (size x size uint8) risk grid, nearest-upscaled, clipped to the
    route corridor -- the exact rendering used for both the saved heatmap/
    PNGs (collect_data_topdown.py) and the live inference overlay
    (heatmap_agent.py), so offline and online heatmaps stay identical.

    The route corridor is always painted at least `corridor_floor` (fraction of
    255) so it shows as "safe" green even with no obstacles, instead of blank.
    Outside the corridor stays 0. With no route_pts there is no corridor.
    """
    if obstacles:
        img = render_ego_bev_image(
            obstacles, ego_xy, ego_yaw,
            x_range=(-topdown_half, topdown_half),
            y_range=(-topdown_half, topdown_half),
            resolution=resolution,
            colormap=False,           # raw grayscale risk (0-255)
            horizon_scale=horizon_scale,
        )
        if img.shape[:2] != (size, size):
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
        arr = img.copy()
    else:
        arr = np.zeros((size, size), dtype=np.uint8)

    if window == 'cone':
        mask = reachable_cone_mask(ego_speed, topdown_half, size)
    elif route_pts and len(route_pts) >= 2:
        heatmap_route = clip_route_to_bumper(route_pts)
        mask = route_corridor_mask(heatmap_route, corridor_half_width, topdown_half, size)
    else:
        mask = None

    if mask is not None:
        arr[~mask] = 0
        floor = int(round(corridor_floor * 255))
        if floor > 0:
            arr[mask & (arr < floor)] = floor

    # Static obstacles (parked cars): punch a hole in the heatmap at each parked
    # car so it flows AROUND them instead of painting over them. Applied last so
    # it overrides the floor; carved cells go to 0 (untinted, like outside the
    # window) -- no risk colour added for static cars, just a cut-out.
    if static_boxes:
        cutout = static_obstacle_mask(static_boxes, size, topdown_half)
        arr[cutout] = 0
    return arr


def overlay_on_topdown(obstacles, ego_xy, ego_yaw, topdown_img,
                       height=18.0, fov=90.0, alpha=0.5, risk_floor=0.05,
                       flip_lr=False, flip_ud=False, **snapshot_kw):
    """Blend the ego danger-zone heatmap onto a top-down camera frame."""
    size = topdown_img.shape[0]
    half = height * np.tan(np.deg2rad(fov / 2.0))
    res  = (2.0 * half) / size

    risk, _ = render_ego_bev(obstacles, ego_xy, ego_yaw,
                             x_range=(-half, half), y_range=(-half, half),
                             resolution=res, **snapshot_kw)
    hm = to_forward_up_image(risk)
    if hm.shape[:2] != (size, size):
        hm = cv2.resize(hm, (size, size), interpolation=cv2.INTER_LINEAR)
    if flip_lr:
        hm = hm[:, ::-1]
    if flip_ud:
        hm = hm[::-1, :]

    # Use soft green/red instead of JET.
    color = colorize_soft(hm)                  # BGR already
    mask  = (hm >= risk_floor)[..., None]
    out   = topdown_img.copy()
    blended = (1.0 - alpha) * out + alpha * color
    return np.where(mask, blended, out).astype(np.uint8)


def _letterbox_square(img, side):
    """Aspect-preserving resize of img into a side x side canvas (zero-padded).

    Used so neither view is geometrically distorted when InternVL2 later tiles
    the stitched image into square 448x448 patches -- a wide (2:1) front camera
    forced straight into a square tile is what produced the squashed look.
    """
    h, w = img.shape[:2]
    scale = side / float(max(h, w))
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((side, side, 3), dtype=img.dtype)
    y0, x0 = (side - nh) // 2, (side - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def heatmap_on_topdown(topdown_img, risk, risk_floor=0.05):
    """Tint topdown_img with the corridor-clipped risk heatmap (soft green/red).

    risk: aligned with topdown_img, float [0,1] or uint8 grayscale (heatmap PNG).
    Only pixels with risk >= risk_floor are tinted; everywhere else (e.g. outside
    the route corridor, already zeroed by route_corridor_mask) keeps the original
    topdown pixels. Same risk_floor/np.where gating as overlay_on_topdown.
    """
    risk = np.asarray(risk, dtype=np.float32)
    if risk.max() > 1.0:
        risk = risk / 255.0
    color_bgr = colorize_soft(risk)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    alpha = (0.35 + 0.50 * np.clip(risk * 3.0, 0.0, 1.0))[..., np.newaxis]
    blended = np.clip(
        topdown_img.astype(np.float32) * (1.0 - alpha) + color_rgb * alpha,
        0, 255).astype(np.uint8)
    mask = (risk >= risk_floor)[..., None]
    return np.where(mask, blended, topdown_img).astype(np.uint8)


def bake_heatmap_overlay(rgb_img, topdown_img, risk, mode='ego_topdown', risk_floor=0.05):
    """Build the exact RGB image fed to the model.

    rgb_img, topdown_img: RGB uint8 arrays (front camera, topdown camera).
    risk: risk array aligned with topdown_img (float [0,1] or uint8 grayscale).

    mode='ego_topdown' -> [front camera | topdown+heatmap] as two equal squares
        side-by-side (2:1). Each view is letterboxed into its own square so that
        InternVL2's 2-patch tiling lands one undistorted view per tile.
    mode='topdown_only' -> just the square topdown+heatmap (no ego view).

    The result is written into rgb/ in place (data collection) or returned to the
    agent at inference -- the dataloader reads it as-is, no simlingo changes.
    """
    td = heatmap_on_topdown(topdown_img, risk, risk_floor)   # square (e.g. 512x512)
    if mode == 'topdown_only':
        return td
    if mode != 'ego_topdown':
        raise ValueError(f"unknown bake mode: {mode!r}")
    side = td.shape[0]
    ego_sq = _letterbox_square(rgb_img, side)
    return np.hstack([ego_sq, td])


if __name__ == "__main__":
    obstacles = [{
        'edge_point': [3.0, 1.5], 'cross_dir': [0.0, 1.0],
        'cross_distance': 6.5, 'speed': 1.3, 'p_cross': 0.5,
        'extent': [0.3, 0.3],
    }]
    img = render_ego_bev_image(obstacles, ego_xy=[0.0, 0.0], ego_yaw=0.0)
    print("heatmap image:", img.shape, "dtype", img.dtype, "max", int(img.max()))