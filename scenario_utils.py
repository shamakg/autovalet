"""
Shared collision-tracking and scoring utilities used by both default_runner.py
and vla_adapter/benchmark.py. Import from here to keep the two runners in sync.
"""

import numpy as np
import carla

from testbed.v2_experiment_utils import near_miss, obstacle_map_from_bbs

GRACE_TICKS = 20  # ~1 s at 20 Hz


def calculate_min_distance_to_door(ego_actor, door_vehicle):
    """Minimum distance between any ego corner and either front door segment."""
    ego_trans = ego_actor.get_transform()
    ego_ext = ego_actor.bounding_box.extent
    ego_corners_local = [
        carla.Location(x=-ego_ext.x, y=-ego_ext.y, z=0),
        carla.Location(x=-ego_ext.x, y= ego_ext.y, z=0),
        carla.Location(x= ego_ext.x, y=-ego_ext.y, z=0),
        carla.Location(x= ego_ext.x, y= ego_ext.y, z=0),
    ]
    ego_p = [np.array([ego_trans.transform(c).x, ego_trans.transform(c).y]) for c in ego_corners_local]

    door_trans = door_vehicle.get_transform()
    door_ext = door_vehicle.bounding_box.extent
    door_segments = []
    for side in [-1, 1]:
        h_loc = carla.Location(x=0.5*door_ext.x, y=side*door_ext.y, z=0)
        e_loc = carla.Location(x=0.5*door_ext.x, y=side*(door_ext.y + 1.0), z=0)
        h_world = door_trans.transform(h_loc)
        e_world = door_trans.transform(e_loc)
        door_segments.append((np.array([h_world.x, h_world.y]), np.array([e_world.x, e_world.y])))

    ego_yaw_rad = np.deg2rad(ego_trans.rotation.yaw)
    cos_a = np.cos(-ego_yaw_rad)
    sin_a = np.sin(-ego_yaw_rad)

    min_dist = float('inf')
    for s1, s2 in door_segments:
        for pt in [s1, s2]:
            dx = pt[0] - ego_trans.location.x
            dy = pt[1] - ego_trans.location.y
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            if abs(lx) <= ego_ext.x and abs(ly) <= ego_ext.y:
                return 0.0
        for p in ego_p:
            min_dist = min(min_dist, _dist_point_to_segment(p, s1, s2))
        ego_edges = [(ego_p[0], ego_p[1]), (ego_p[1], ego_p[3]), (ego_p[3], ego_p[2]), (ego_p[2], ego_p[0])]
        for ep1, ep2 in ego_edges:
            min_dist = min(min_dist, _dist_point_to_segment(s1, ep1, ep2))
            min_dist = min(min_dist, _dist_point_to_segment(s2, ep1, ep2))
    return min_dist


def _dist_point_to_segment(p, s1, s2) -> float:
    v = s2 - s1
    w = p - s1
    c1 = np.dot(w, v)
    if c1 <= 0:
        return float(np.linalg.norm(p - s1))
    c2 = np.dot(v, v)
    if c2 <= c1:
        return float(np.linalg.norm(p - s2))
    return float(np.linalg.norm(p - (s1 + (c1 / c2) * v)))


def compute_weighted_iou(iou, is_door_mode, min_door_dist=float('inf')):
    """Return IOU * 0.5 + door_safety * 0.5 for DoorMode; otherwise return iou unchanged.

    Door safety score: 0 if dist < 0.05 m, 1 if dist > 0.30 m, linear in between.
    """
    if not is_door_mode:
        return iou
    safety_score = float(np.clip((min_door_dist - 0.05) / (0.30 - 0.05), 0.0, 1.0))
    weighted = iou * 0.5 + safety_score * 0.5
    print(f'Door Safety Min Dist: {min_door_dist:.3f}m, Safety Score: {safety_score:.3f}')
    print(f'Weighted IOU: {weighted:.3f}')
    return weighted


def obb_aabb_overlap(cur, front_m, rear_m, half_w, aabb):
    """Exact OBB-AABB overlap via Separating Axis Theorem. No grid, no quantization."""
    wx  = (aabb[0] + aabb[2]) / 2
    wy  = (aabb[1] + aabb[3]) / 2
    wdx = (aabb[2] - aabb[0]) / 2
    wdy = (aabb[3] - aabb[1]) / 2
    cos_a, sin_a = np.cos(cur.angle), np.sin(cur.angle)
    half_len = (front_m + rear_m) / 2
    obb_cx = cur.x + (front_m - rear_m) / 2 * cos_a
    obb_cy = cur.y + (front_m - rear_m) / 2 * sin_a
    dx, dy = wx - obb_cx, wy - obb_cy
    for ax, ay in [(1, 0), (0, 1), (cos_a, sin_a), (-sin_a, cos_a)]:
        obb_proj  = abs(half_len * (ax*cos_a + ay*sin_a)) + abs(half_w * (-ax*sin_a + ay*cos_a))
        aabb_proj = abs(wdx * ax) + abs(wdy * ay)
        if abs(dx*ax + dy*ay) > obb_proj + aabb_proj:
            return False
    return True


def analyze_scenario(parking_scenario):
    """Return (result_str, num_collisions) for the completed scenario."""
    result = "SUCCESS"
    print("EVENTS:")
    num_collisions = 0
    for criterion in parking_scenario.get_criteria():
        criterion.terminate(None)
        if hasattr(criterion, 'actual_value'):
            num_collisions += criterion.actual_value
        print(criterion.name, criterion.test_status)
        if criterion.test_status != "SUCCESS":
            result = "SCENARIO FAILURE"
    print("FINAL SCENARIO RESULT:", result)
    print("ACTUAL NUMBER OF COLLISIONS:", num_collisions)
    return result, num_collisions


def draw_car(parking_scenario, world):
    car_transform = parking_scenario.car.actor.get_transform()
    loc = car_transform.location

    car = parking_scenario.car.car
    length_front = car.front_m
    length_rear  = car.rear_m
    width        = car.half_width_m * 2
    cur = car.cur
    yaw = cur.angle

    local_corners = np.array([
        [-length_rear, -width/2],
        [-length_rear,  width/2],
        [ length_front,  width/2],
        [ length_front, -width/2],
    ])
    R = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)],
    ])
    world_corners = [np.dot(R, corner) + np.array([cur.x, cur.y]) for corner in local_corners]

    for i in range(4):
        p1 = world_corners[i]
        p2 = world_corners[(i+1) % 4]
        world.debug.draw_line(
            carla.Location(x=p1[0], y=p1[1], z=loc.z + 0.1),
            carla.Location(x=p2[0], y=p2[1], z=loc.z + 0.1),
            thickness=0.5,
            color=carla.Color(r=255, g=0, b=0),
            life_time=1,
            persistent_lines=True,
        )


def track_collisions(
    parking_scenario,
    collision_tracker,
    traffic_cone_bbs,
    walker_bbs,
    vehicle_criterion,
    prev_vehicle_count,
    colliding_walker_ids,
    walker_last_seen,
    near_miss_active,
    near_miss_min_distance,
    collisions_ref,
    walker_collisions_ref,
    near_misses_ref,
    near_miss_threshold,
    cone_obs_cached=None,
):
    """
    One tick of collision and near-miss tracking.

    Mutates colliding_walker_ids and walker_last_seen in-place.
    Returns (prev_vehicle_count, near_miss_active, near_miss_min_distance).

    cone_obs_cached: pre-computed dilated cone obstacle map (pass once before the
    loop for static cones; None falls back to rebuilding it each tick).
    """
    # Vehicle collision: read CollisionTest criterion
    if vehicle_criterion is not None:
        new_count = vehicle_criterion.actual_value
        if new_count > prev_vehicle_count:
            collisions_ref[0] += new_count - prev_vehicle_count
            print(f"Vehicle Collision Detected! (total so far: {new_count})")
            print("Current car location", parking_scenario.car.actor.get_location())
        prev_vehicle_count = new_count

    # Car footprint mask on the collision grid
    car = parking_scenario.car.car
    collision_mask = collision_tracker.generate_collision_mask(
        car.cur, front_m=car.front_m, rear_m=car.rear_m, half_width_m=car.half_width_m,
    )

    # Per-walker check: exact OBB-AABB SAT — no grid quantization or dilation errors.
    walker_ids_this_tick = set()
    for w_id, w_bb in walker_bbs:
        if obb_aabb_overlap(car.cur, car.front_m, car.rear_m, car.half_width_m, w_bb):
            walker_ids_this_tick.add(w_id)

    # Cones: no per-actor IDs, use sentinel key 'cone'.
    # Use pre-computed dilated map when available (cones are static).
    if cone_obs_cached is not None:
        if np.any(collision_mask & (cone_obs_cached == 1)):
            walker_ids_this_tick.add('cone')
    elif traffic_cone_bbs:
        cone_obs = obstacle_map_from_bbs(traffic_cone_bbs, collision_tracker).obs
        cone_obs[0, :] = 0; cone_obs[-1, :] = 0
        cone_obs[:, 0] = 0; cone_obs[:, -1] = 0
        _c = cone_obs
        cone_obs = _c | np.roll(_c, 1, 0) | np.roll(_c, -1, 0) | np.roll(_c, 1, 1) | np.roll(_c, -1, 1)
        if np.any(collision_mask & (cone_obs == 1)):
            walker_ids_this_tick.add('cone')

    # Hysteresis: keep an ID in the active set for GRACE_TICKS ticks after the
    # last detected overlap so grid-quantization flicker doesn't re-count the
    # same crossing event.
    for wid in walker_ids_this_tick:
        walker_last_seen[wid] = 0
    for wid in list(walker_last_seen.keys()):
        if wid not in walker_ids_this_tick:
            walker_last_seen[wid] += 1
            if walker_last_seen[wid] > GRACE_TICKS:
                colliding_walker_ids.discard(wid)
                del walker_last_seen[wid]

    new_collisions = walker_ids_this_tick - colliding_walker_ids
    if new_collisions:
        walker_collisions_ref[0] += len(new_collisions)
        print(f"Walker/Cone Collision(s) Detected: {new_collisions}")
    colliding_walker_ids |= walker_ids_this_tick

    # Near-miss: reset during active collision so that a near-miss→collision→safe
    # transition does not produce a spurious near-miss count on exit.
    if walker_ids_this_tick:
        near_miss_active = False
    all_walker_bbs = [bb for _, bb in walker_bbs] + traffic_cone_bbs
    if not walker_ids_this_tick and all_walker_bbs:
        combined_obs = obstacle_map_from_bbs(all_walker_bbs, collision_tracker).obs
        combined_obs[0, :] = 0; combined_obs[-1, :] = 0
        combined_obs[:, 0] = 0; combined_obs[:, -1] = 0
        _w2 = combined_obs
        combined_obs = _w2 | np.roll(_w2, 1, 0) | np.roll(_w2, -1, 0) | np.roll(_w2, 1, 1) | np.roll(_w2, -1, 1)
        distance = near_miss(collision_mask, combined_obs)
        if 0 < distance < near_miss_threshold:
            near_miss_min_distance = min(near_miss_min_distance, distance) if near_miss_active else distance
            near_miss_active = True
        else:
            if near_miss_active:
                near_misses_ref[0] += 1
                print(f"Near-miss! Closest distance: {near_miss_min_distance:.2f}m")
            near_miss_active = False
    elif not walker_ids_this_tick:
        near_miss_active = False

    return prev_vehicle_count, near_miss_active, near_miss_min_distance
