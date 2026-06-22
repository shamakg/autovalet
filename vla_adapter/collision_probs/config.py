# Ego vehicle footprint — Lincoln MKZ dimensions, matching ObstacleMap.generate_collision_mask.
EGO_HALF_LENGTH = (3.856 + 1.045) / 2.0   # ~2.45 m
EGO_HALF_WIDTH  = 1.09                      # m

# Fallback obstacle footprint when a GMM mode carries no extent info.
DEF_OBS_HALF_LENGTH = 2.3   # m
DEF_OBS_HALF_WIDTH  = 1.0   # m

# GMM sweep parameters (build_crossing_snapshots defaults)
HORIZON_SCALE = 2.5    # project obstacles this far beyond cross_distance (1.0 = lane only)
TIME_HORIZON  = 8.0    # total sweep window (seconds)
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

# --- Reachability-cone heatmap window (bicycle model) -----------------------
# Alternative to the route corridor: instead of masking risk to a tube around the
# A* route (which hands the model its exact path -> shortcut that suppressed the
# swing-out), mask to the ego's forward REACHABLE SET under a kinematic bicycle
# model -- the fan of positions it could drive to, bounded laterally by the
# minimum turning radius and longitudinally by speed*reach_time. The model then
# sees "where I could go + where the danger is", not "the line to follow".
VEHICLE_WHEELBASE = 2.8     # m  -- ego wheelbase
MAX_STEER_DEG     = 35.0    # deg -- effective max road-wheel angle -> min turn radius
CONE_REACH        = 10.0    # m  -- FIXED cone length each way (forward + reverse),
                            #       independent of speed. Raise for a longer cone.
CONE_HALF_ANGLE_DEG = 22.0  # deg -- half-angle of the fan. The reachable half-width
                            #        opens linearly as base_half + tan(angle)*|x|, so
                            #        the window reads as a true cone (straight angled
                            #        sides) instead of the min-turn-radius arc that
                            #        saturated into square sides. Widen to open it more.

# --- Static obstacles (parked cars) ------------------------------------------
# The crossing-GMM risk only models DYNAMIC obstacles (walkers/vehicles that
# cross). Parked cars were invisible to the heatmap, so the reachable cone
# flooded straight over them. Using the per-frame static boxes (from boxes/, ego
# frame, speed~0) we punch a HOLE in the heatmap at each parked car so the green
# safe region flows around them instead of painting over them. No risk colour is
# added for static cars -- they are simply cut out.
STATIC_OBSTACLES_ENABLED = True
STATIC_EGO_INFLATE_M     = 0.0   # m -- small clearance added around each car's
                                  # footprint. The hole hugs the actual car; raise
                                  # toward EGO_HALF_WIDTH (1.09) for ego-centre
                                  # clearance, set 0 for the bare car outline.
STATIC_SPEED_EPS         = 0.5    # m/s -- treat boxes slower than this as static.

# --- Rebake-friendly storage -------------------------------------------------
# Keep the clean topdown/ BEV frames after collection instead of deleting them.
# topdown/ is the ONLY non-regenerable input the heatmap overlay needs (the
# overlay is alpha-blended INTO topdown_heatmap/, so it can't be inverted back to
# a clean base). Keeping it lets rebake.py regenerate topdown_heatmap/ + heatmap/
# + risk_grid/ offline for any heatmap change -- no CARLA re-collection ever.
# Those three derived dirs become a disposable cache (rebake.py --prune-cache
# reclaims them; rebake.py --all rebuilds them). heatmap/ (grayscale) is still
# deleted at collection time since it's cheap to recompute. Net cost: ~one extra
# BEV jpg per frame, offset by being able to prune the derived cache when idle.
KEEP_TOPDOWN_BASE = True

# --- Collision-loss risk grid ------------------------------------------------
# Per-frame UN-masked obstacle risk grid dumped alongside topdown_heatmap/ (in
# risk_grid/) and sampled under the predicted waypoints by the auxiliary
# ChauffeurNet-style collision loss. Un-masked (no cone/corridor clip) so the
# loss can penalise predictions that leave the safe window into an obstacle.
RISK_GRID_SIZE = 96    # px (ego forward-up, same geometry as the heatmap window)
