### Data Structure (From dataset_base.py)
database/parking/data/simlingo/your_run/routes_training/RouteScenario_X/TownXX/
    rgb/
        0000.jpg
        0001.jpg
        ...
    measurements/
        0000.json.gz       ← gzipped JSON, not plain JSON
        0001.json.gz
    boxes/
        0000.json.gz
    results.json.gz        ← route pass/fail file


Add a custom parking plan to lmdrive.json, add the new key to command_template_mappings in dataset_base.py. 
     2. simlingo_training/dataloader/dataset_base.py — add to both dicts:
  map_command = {
      ...
      7: 'park in the designated spot',   # new
  }
  command_template_mappings = {
      ...
      7: [65],   # points to your new lmdrive.json key
  }
  

### Randomize starting positions
- Randomize the position in the entrance of the lane
- For recovery data, we can randomize position throughout the lane (this only works for pedestrian and cone scenarios)

To begin with, lets just consider:
Pedestrian Scenario 
Cone Scenario
Possible Door Scenario

Randomize scenarios

Boxes: bbs of nearby actors in ego-relative coordinates, saved per frame. Save parked cars and walkers

Use the Kalman Filter Hybrid A star approach as the expert trajectory, filtering out scenarios that end in failure (collision or parking failure)

Inputs:
    - Route: Simply the target point (destination/center of the parking spot)
        - Made sure to change agent_interface to agree with this

---

## How collect_data.py processes the hybrid A* trajectory into the route label

The hybrid A* planner (v2.py `plan_hybrid_a_star`) runs inside `Car.plan()` and produces
`car.car.trajectory`: a list of `TrajectoryPoint` objects in world coordinates, densely
sampled along the planned path from the current position to the destination.

Each saved frame calls `save_measurement`, which converts that trajectory into the 20-waypoint
`route` field stored in the measurement JSON:

### Step 1 — find the closest trajectory point to the car center
```
traj_xy = [[wp.x, wp.y] for wp in trajectory]
ti = argmin(||traj_xy[i] - ego_xy||)
forward_traj = trajectory[ti:]
```
`car.car.ti` is NOT used here — it tracks the rear-axle reference (~1.6 m behind center) and
would place route[0] behind the car. Instead we find the closest point by Euclidean distance.

### Step 2 — anchor route[0] at the car center
```
route[0] = [0.0, 0.0]   # always exactly the car's center in ego frame
```
This ensures the route label always starts at the ego origin regardless of where the closest
trajectory point fell. It avoids a training/inference mismatch where the model sees route[0]
slightly behind the car.

### Step 3 — sample 19 more points from forward_traj[1:]
```
rest_traj = forward_traj[1:]          # skip the closest point (already replaced by [0,0])
indices   = linspace(0, len-1, 19)    # evenly spaced by index
route[1:] = [ego_frame(rest_traj[i]) for i in indices]
```
Sampling starts from `forward_traj[1:]` (strictly after the closest point) so route[1] never
lands behind route[0]. Sampling is uniform by waypoint index, NOT by metric distance — meaning
spacing in metres is roughly uniform only if the A* waypoints are evenly spaced (they generally
are for the hybrid A* grid step size used here).

### Coordinate frame
All route waypoints are converted to ego frame via `inverse_conversion_2d`:
- X axis: forward (direction the car is facing)
- Y axis: left

For reverse maneuvers, forward_traj[1:] naturally contains negative-X waypoints (behind the
car in the facing direction), which is correct — those represent the planned reverse path.

### Known issues / history
- **run_001 bug**: the original code used `linspace(0, len-1, 20)` from `forward_traj[0:]`
  (no route[0]=[0,0] anchor, no [1:] skip). This caused route[0] to land up to 1.5 m behind
  the car in ~35% of frames. Fixed in post-processing for run_001 (route[0] forced to [0,0];
  route[1:] left as-is, which is an approximation).
- **run_001 deletion mistake**: all speed<0 frames were deleted from run_001, including 39
  genuine reverse-parking episodes. Future runs use the corrected collect_data.py which does
  not filter by speed.

Data:
Some things I noticed: for the open door scenarios, the car stops prematurely

After first round of training
    - 

Figure Idea
    Pointcloud of start parking positions


### May 28: Reread the paper, and realized we need bucketing
# Bucket 1: Straight approach (boring, downsample)
route lateral excursion < 1.0m AND target_point[0] > 10.0

# Bucket 2: Swing-out phase (critical, upsample 3-5x)
route lateral excursion > 2.0m AND target_point[0] > 3.0

# Bucket 3: Final turn into spot (critical, upsample 3-5x)  
target_point[0] < 3.0

# Bucket 4: Recovery frames (important, upsample 2x)
route[0][0] < -0.5

## Also Adding Img shift augmentation


### New Train/test split:
TRAIN_SCENARIOS = [
    (16, [15, 17]),  # row2 col1
    (17, [16, 18]),  # row2 col2
    (18, [17, 19]),  # row2 col3
    (19, [18, 20]),  # row2 col4
    (20, [19, 21]),  # row2 col5
    (21, [20, 22]),  # row2 col6
    (22, [21, 23]),  # row2 col7
    (23, [22, 24]),  # row2 col8
    (24, [23, 25]),  # row2 col9
    (25, [24, 26]),  # row2 col10
    (26, [25, 27]),  # row2 col11
    (29, [28, 30]),  # row2 col14 ← far hint
    (32, [31, 33]),  # row3 col1
    (33, [32, 34]),  # row3 col2
    (34, [33, 35]),  # row3 col3
    (35, [34, 36]),  # row3 col4
    (36, [35, 37]),  # row3 col5
    (37, [36, 38]),  # row3 col6
    (38, [37, 39]),  # row3 col7
    (39, [38, 40]),  # row3 col8
    (40, [39, 41]),  # row3 col9
    (41, [40, 42]),  # row3 col10
    (42, [41, 43]),  # row3 col11
    (45, [44, 46]),  # row3 col14 ← far hint
]

SCENARIOS = [
    (27, [26, 28]),  # row2 col12
    (28, [27, 29]),  # row2 col13
    (30, [29, 31]),  # row2 col15
    (31, [30, 32]),  # row2 col16
    (43, [42, 44]),  # row3 col12
    (44, [43, 45]),  # row3 col13
    (46, [45, 47]),  # row3 col15
    (47, [46, 48]),  # row3 col16
]


       Parameter       │   simlingo source    │ Our value │ Status  │
  ├───────────────────────┼──────────────────────┼───────────┼─────────┤
  │ Camera position       │ [-1.5, 0.0, 2.0]     │ same      │ ✓       │
  ├───────────────────────────┼────────────────────────────────────────┼───────────┼─────────┤
  │ Camera rotation           │ [0, 0, 0] deg                          │ same      │ ✓       │
  ├───────────────────────────┼────────────────────────────────────────┼───────────┼─────────┤
  │ Image size                │ 1024 × 512                             │ same      │ ✓       │
  ├───────────────────────────┼────────────────────────────────────────┼──────────────┼─────────┤
  │ FOV                       │ 110                                    │ same         │ ✓       │
  ├───────────────────────────┼────────────────────────────────────────┼──────────────┼─────────┤
  │ aug_translation range     │ uniform(-1.5, 1.5) m                   │ same         │ ✓ fixed │
  ├───────────────────────────┼────────────────────────────────────────┼──────────────┼─────────┤
  │ aug_rotation range        │ uniform(-20, 20) deg                   │ same         │ ✓ fixed │
  ├─────────────────────────────┼────────────────────────────────────────┼──────────────┼─────────┤
  │ aug_rotation storage unit   │ degrees (dataset does deg2rad on load) │ degrees      │ ✓ fixed │
  ├─────────────────────────────┼────────────────────────────────────────┼───────────────┼─────────┤
  │ SAVE_EVERY_N                │ data_save_freq = 5                     │ 5             │ ✓       │
  ├─────────────────────────────┼────────────────────────────────────────┼─────────────────┼─────────┤
  │ skip_first_n_frames         │ int(2.5×20)//5 = 10                    │ 10 (default)    │ ✓       │
  ├─────────────────────────────┼────────────────────────────────────────┼─────────────────┼─────────┤
  │ pred_len                    │ 11                                     │ 11              │ ✓       │
  ├─────────────────────────────┼────────────────────────────────────────┼─────────────────┼─────────┤
  │ hist_len                    │ 1                                      │ 1               │ ✓       │
  ├─────────────────────────────┼────────────────────────────────────────┼─────────────────┼─────────┤
  │ img_shift_augmentation_prob │ 0.5                                    │ 0.5             │ ✓       │
  ├─────────────────────────────┼────────────────────────────────────────┼─────────────────┼─────────┤
  │ LoRA α / r / dropout        │ 64 / 32 / 0.1                          │ 64 / 32 / 0.1   │ ✓       │
  ├─────────────────────────────┼────────────────────────────────────────┼─────────────────┼─────────┤
  │ weight_decay                │ 0.1                                    │ 0.1 (inherited) │ ✓       │
  ├─────────────────────────────┼────────────────────────────────────────┼─────────────────┼─────────┤
  │ betas                       │ (0.9, 0.999)                           │ inherited       │ ✓       │
  ├─────────────────────────────┼────────────────────────────────────────┼─────────────────┼─────────┤
  │ pct_start                   │ 0.05                                   │ inherited       │ ✓      

Important Design Decision:
When the car stops for pedestrians, I decided to cluster all the way points at the ego origin (I think this models what a traditional expert would do)


Potentially;
Add Random Noise