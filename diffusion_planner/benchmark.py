"""benchmark.py — Diffusion vs Hybrid A* benchmark runner.

Usage:
    Set PLANNER = 'hybrid_a_star' or 'diffusion' and run from the autovalet directory:
        cd leaderboard/leaderboard/autovalet
        python diffusion_planner/benchmark.py

Results are saved to benchmark_results.pkl.

Phase 1: synchronous, no latency modeling for the planner.
Phase 2 (future): add event queue for diffusion inference time, same pattern as
    perception events here.
"""

from __future__ import annotations

import os
import sys
import pickle
from enum import Enum

import numpy as np

# Allow imports from parent autovalet directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import carla
from v2_experiment_utils import (
    approximate_bb_from_center,
    clear_destination_obstacle_map,
    DEBUG,
    EGO_VEHICLE,
    is_done,
    load_client,
    ms_to_ticks,
    near_miss,
    obstacle_map_from_bbs,
    town04_load,
    town04_spectator_bev,
    town04_spawn_parked_cars,
)
from parking_position import (
    parking_vehicle_locations_Town04,
    player_location_Town04,
)

import v2_diffusion
from v2_diffusion import CarlaCar, Mode
from v2_experiment import SCENARIOS

# ---------------------------------------------------------------------------
# Config — single flag to switch planners
# ---------------------------------------------------------------------------
PLANNER = 'hybrid_a_star'    # 'hybrid_a_star' or 'diffusion'
PLANNING_PERIOD = 500         # ms — both planners called at same frequency
PERCEPTION_PERIOD = 200       # ms
DATA_COLLECTION_PERIOD = 200  # ms
TIMEOUT = 120 * 1000          # ms
NUM_RANDOM_CARS = 50
PERCEPTION_LATENCY = 0        # ms — Phase 1: synchronous, no latency

NEAR_MISS_THRESHOLD = 2.0     # metres

# Ego vehicle bounding box (Audi e-tron)
FRONT_M      = 3.856
REAR_M       = 1.045
HALF_WIDTH_M = 1.09

# ---------------------------------------------------------------------------
# Collision helpers (copied from runner_test_medium.py)
# ---------------------------------------------------------------------------

def obb_aabb_overlap(cur, front_m, rear_m, half_w, aabb):
    """Exact OBB-AABB overlap via Separating Axis Theorem."""
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


# ---------------------------------------------------------------------------
# Event queue (same pattern as runner_test_medium.py)
# ---------------------------------------------------------------------------

class EventType(Enum):
    PERCEPTION = 1


class Event:
    def __init__(self, event_type: EventType, time: int, data):
        self.type = event_type
        self.time = time
        self.data = data


# ---------------------------------------------------------------------------
# Ego vehicle spawn using v2_diffusion.CarlaCar
# ---------------------------------------------------------------------------

def _spawn_ego_vehicle(world, destination_parking_spot):
    destination_loc = parking_vehicle_locations_Town04[destination_parking_spot]
    blueprint = world.get_blueprint_library().filter(EGO_VEHICLE)[0]
    return CarlaCar(
        world,
        blueprint,
        player_location_Town04,
        destination_loc,
        approximate_bb_from_center(destination_loc),
        debug=DEBUG,
    )


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def run_scenario(world, destination_parking_spot, parked_spots):
    parked_cars = []
    car = None
    try:
        parked_cars, parked_cars_bbs, parked_cars_and_spots_bbs = town04_spawn_parked_cars(
            world, parked_spots, destination_parking_spot, NUM_RANDOM_CARS
        )
        car = _spawn_ego_vehicle(world, destination_parking_spot)

        # Ground-truth obstacle map (same as v2_experiment.py "perfect perception" hack)
        car.car.obs = obstacle_map_from_bbs(parked_cars_and_spots_bbs)
        clear_destination_obstacle_map(car.car.obs, destination_parking_spot)

        # Separate ground-truth map used only for near-miss distance computation
        collision_tracker = obstacle_map_from_bbs(parked_cars_bbs)

        collisions = [0]
        colliding_car_ids = set()   # IDs currently in collision
        car_last_seen = {}           # {id: ticks_since_last_overlap}
        GRACE_TICKS = 20
        near_miss_active = False
        near_miss_min_distance = float('inf')
        near_miss_count = 0
        locations = []
        events = []
        perception_delay = ms_to_ticks(PERCEPTION_LATENCY)

        i = 0
        while not is_done(car):
            world.tick()
            car.localize()

            cur = car.car.cur

            # --- Parked car collision: exact OBB-AABB, per-car debounce ---
            car_ids_this_tick = set()
            for idx, bb in enumerate(parked_cars_bbs):
                if obb_aabb_overlap(cur, FRONT_M, REAR_M, HALF_WIDTH_M, bb):
                    car_ids_this_tick.add(idx)

            for cid in car_ids_this_tick:
                car_last_seen[cid] = 0
            for cid in list(car_last_seen.keys()):
                if cid not in car_ids_this_tick:
                    car_last_seen[cid] += 1
                    if car_last_seen[cid] > GRACE_TICKS:
                        colliding_car_ids.discard(cid)
                        del car_last_seen[cid]

            new_collisions = car_ids_this_tick - colliding_car_ids
            if new_collisions:
                collisions[0] += len(new_collisions)
                print(f'Collision with parked car(s): {new_collisions}')
            colliding_car_ids |= car_ids_this_tick

            # --- Near-miss (only when no active collision) ---
            if not car_ids_this_tick:
                collision_mask = collision_tracker.generate_collision_mask(cur)
                gt_obs = collision_tracker.obs
                distance = near_miss(collision_mask, gt_obs)
                if 0 < distance < NEAR_MISS_THRESHOLD:
                    if near_miss_active:
                        near_miss_min_distance = min(near_miss_min_distance, distance)
                    else:
                        near_miss_min_distance = distance
                    near_miss_active = True
                else:
                    if near_miss_active:
                        near_miss_count += 1
                        print(f'Near-miss! closest: {near_miss_min_distance:.2f}m')
                    near_miss_active = False
            else:
                near_miss_active = False

            # Data collection every DATA_COLLECTION_PERIOD
            if i % ms_to_ticks(DATA_COLLECTION_PERIOD) == 0:
                locations.append((cur.x, cur.y))

            # Perception event queue
            if i % ms_to_ticks(PERCEPTION_PERIOD) == 0:
                imgs = car.car.camera_sensor.get_images()
                cur = car.car.cur
                events.append(Event(
                    EventType.PERCEPTION,
                    i + perception_delay,
                    (cur.x, cur.y, cur.angle, imgs),
                ))
            for event in list(events):
                if i == event.time and event.type == EventType.PERCEPTION:
                    car.perceive(*event.data)
                    clear_destination_obstacle_map(car.car.obs, destination_parking_spot)

            # Plan at fixed period — no collision trigger
            if i % ms_to_ticks(PLANNING_PERIOD) == 0:
                car.plan()

            car.run_step()

            if i > ms_to_ticks(TIMEOUT):
                car.fail()

            i += 1

        iou = car.iou()
        return {
            'scenario': (destination_parking_spot, parked_spots),
            'iou': iou,
            'collisions': collisions[0],
            'near_misses': near_miss_count,
            'time_to_park': i,
            'success': car.car.mode == Mode.PARKED,
            'locations': locations,
        }

    finally:
        if car is not None:
            car.destroy()
        for parked_car in parked_cars:
            parked_car.destroy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Apply planner selection before any Car objects are created
    v2_diffusion.USE_DIFFUSION = (PLANNER == 'diffusion')
    planner_name = PLANNER

    results = {planner_name: []}

    try:
        client = load_client()
        world = town04_load(client)
        town04_spectator_bev(world)

        for i, (destination, parked_spots) in enumerate(SCENARIOS):
            print(
                f'\n[{planner_name}] Scenario {i + 1}/{len(SCENARIOS)}: '
                f'destination={destination}, parked_spots={parked_spots}'
            )
            result = run_scenario(world, destination, parked_spots)
            results[planner_name].append(result)
            print(
                f'  IoU={result["iou"]:.3f}  '
                f'collisions={result["collisions"]}  '
                f'near_misses={result["near_misses"]}  '
                f'success={result["success"]}  '
                f'ticks={result["time_to_park"]}'
            )

    except KeyboardInterrupt:
        print('stopping simulation')

    finally:
        with open('benchmark_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        print('\nResults saved to benchmark_results.pkl')

        r = results[planner_name]
        if r:
            mean_iou = np.mean([x['iou'] for x in r])
            collision_rate = np.mean([x['collisions'] > 0 for x in r])
            mean_near_misses = np.mean([x['near_misses'] for x in r])
            success_rate = np.mean([x['success'] for x in r])
            mean_ticks = np.mean([x['time_to_park'] for x in r])
            print(f'\n=== {planner_name} summary ({len(r)} scenarios) ===')
            print(f'  Mean IoU:         {mean_iou:.3f}')
            print(f'  Collision rate:   {collision_rate:.3f}')
            print(f'  Mean near-misses: {mean_near_misses:.2f}')
            print(f'  Success rate:     {success_rate:.3f}')
            print(f'  Mean ticks:       {mean_ticks:.0f}')


if __name__ == '__main__':
    main()
