# Ego vehicle footprint — Lincoln MKZ dimensions, matching ObstacleMap.generate_collision_mask.
EGO_HALF_LENGTH = (3.856 + 1.045) / 2.0   # ~2.45 m
EGO_HALF_WIDTH  = 1.09                      # m

# Fallback obstacle footprint when a GMM mode carries no extent info.
DEF_OBS_HALF_LENGTH = 2.3   # m
DEF_OBS_HALF_WIDTH  = 1.0   # m

# GMM sweep parameters (build_crossing_snapshots defaults)
HORIZON_SCALE = 2.5    # project obstacles this far beyond cross_distance (1.0 = lane only)
TIME_HORIZON  = 10.0    # total sweep window (seconds)
N_TIME        = 25     # number of time samples across the window
SIGMA_LON     = 0.25   # longitudinal positional std (m); time-sweep supplies the range
SIGMA_LAT0    = 0.35   # initial lateral std at the edge point (m)
CONE_RATE     = 0.18   # lateral std growth rate (m per m of downrange distance)
STAY_SIGMA    = 0.35   # positional std for STAY (non-crossing) mode (m)

# Top-down camera geometry
TOPDOWN_HEIGHT = 18.0   # camera height above ego (m)
TOPDOWN_SIZE   = 512    # output image size (pixels)
TOPDOWN_FOV    = 90.0   # camera field of view (degrees)

# Rendering / data-collection
# HEATMAP_RADIUS_M          = 12.0   # zero risk beyond this radius from ego centre (m) [legacy circular mask]
ROUTE_CORRIDOR_HALF_WIDTH = 1.5    # zero risk beyond this lateral distance from the A* route centreline (m)

# In-corridor pixels whose risk is below this fraction are raised to it, so the
# route corridor always renders as "safe" green (colorize_soft maps low values to
# green) instead of being blank when there's no obstacle. The model then always
# sees its path: green = clear, amber/red = danger. Outside the corridor stays 0
# (untinted). Set to 0.0 to restore the old blank-when-safe behaviour. Must be
# >= the overlay's risk_floor (0.05 in bake_heatmap_overlay) or the green is gated out.
CORRIDOR_SAFE_FLOOR = 0.1
