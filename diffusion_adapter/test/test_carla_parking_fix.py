"""
test_carla_parking_fix.py  —  CARLA integration test for the L-shaped route guidance fix.

Runs every scenario in v2_experiment.SCENARIOS (currently spots 17 and 18) and asserts
four properties per scenario that were broken before the fix:

  1. AISLE PHASE    — While the car is still in the aisle (CARLA y < AISLE_Y_THRESHOLD),
                      lateral deviation from the spawn x must be < MAX_AISLE_LATERAL_M.
                      The pre-fix car turned diagonally within the first few ticks.

  2. EASTWARD TURN  — Before the scenario ends, the car's x must exceed EAST_THRESHOLD,
                      confirming it turned East into the parking spot.

  3. IOU            — Final parking IOU must exceed MIN_IOU (0.0 means complete failure).

  4. NO COLLISION   — The CARLA VehicleCollisionTest criterion must be 0.

"Full failure" of the model is detected when ALL scenarios return IOU=0, the car
never turns East, and/or collisions occur.  A scenario-level [FAIL] is printed for
each broken assertion; the process exits 1 if any scenario fails.

Run (requires CARLA server on localhost:2000):
    /home/sumesh/envs/simlingo/bin/python test/test_carla_parking_fix.py
"""

import sys, os, time, json
import numpy as np
from datetime import datetime

_HERE         = os.path.dirname(os.path.abspath(__file__))
_ADAPTER_ROOT = os.path.dirname(_HERE)
_AUTOVALET    = os.path.dirname(_ADAPTER_ROOT)
_NUPLAN_ROOT  = os.path.join(_ADAPTER_ROOT, "nuplan-devkit")
_DP_ROOT      = os.path.join(_ADAPTER_ROOT, "Diffusion-Planner")
_CARLA_ROOT   = "/home/sumesh/opt/carla/PythonAPI/carla"
_SCENARIO_ROOT= "/home/sumesh/carla_garage/scenario_runner"
_LB_ROOT      = "/home/sumesh/carla_garage/leaderboard"

for _p in [_CARLA_ROOT, _SCENARIO_ROOT, _LB_ROOT,
           _HERE, _AUTOVALET, _ADAPTER_ROOT, _NUPLAN_ROOT, _DP_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import py_trees
from srunner.scenariomanager.timer import GameTime
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from parking_scenarios.parking_scenario_medium import ParkingScenarioMedium
from v2 import TrajectoryPoint
from v2_experiment import SCENARIOS
from testbed.v2_experiment_utils import load_client, town04_load, town04_spectator_bev
from agent_interface import DiffusionAdapter

# ---- pass/fail thresholds ----
EGO_SPAWN_X         = 285.6    # x at spawn for all Town04 parking scenarios
MAX_AISLE_LATERAL_M = 2.5      # car must stay within this of spawn-x while in the aisle
AISLE_Y_THRESHOLD   = -234.0   # CARLA y below this = car is still in the aisle column
EAST_THRESHOLD      = 289.0    # car must reach at least this x before ending
MIN_IOU             = 0.10     # any positive parking overlap counts as non-failure
WALL_TIMEOUT        = 180      # seconds per scenario

CHECKPOINT_PATH = os.path.join(_ADAPTER_ROOT, "checkpoints/model.pth")


def ok(tag):        print(f"    [PASS] {tag}")
def fail(tag, d=""): print(f"    [FAIL] {tag}  {d}"); return False
def check(cond, tag, detail=""):
    if cond: ok(tag);   return True
    else:    return fail(tag, detail)


# ---------------------------------------------------------------------------

def run_one_scenario(world, destination_spot, parked_spots):
    """
    Run a single parking scenario.
    Returns dict with keys: iou, collisions, positions, ticks.
    """
    adapter          = None
    parking_scenario = None
    actual_positions = []

    try:
        config = ScenarioConfiguration()
        config.name        = "Parking"
        config.type        = "Parking"
        config.town        = "Town04_Opt"
        config.other_actors = None
        config.route       = False

        parking_scenario = ParkingScenarioMedium(
            world=world,
            config=config,
            destination=destination_spot,
            parked=parked_spots,
            debug_mode=0,
            criteria_enable=True,
        )

        world.tick()
        world.tick()

        adapter = DiffusionAdapter()

        dest_raw  = parking_scenario.car.car.destination
        half_len  = parking_scenario.car.actor.bounding_box.extent.x
        front_len = half_len + 1.6
        far_dest  = TrajectoryPoint(
            dest_raw.direction,
            dest_raw.x + front_len * np.cos(dest_raw.angle),
            dest_raw.y + front_len * np.sin(dest_raw.angle),
            dest_raw.speed,
            dest_raw.angle,
        )

        adapter.init_testbed(
            CHECKPOINT_PATH,
            world,
            parking_scenario.car.actor,
            far_dest,
            parking_scenario.car.car.destination.angle,
        )

        vehicle_criterion = next(
            (c for c in parking_scenario.get_criteria()
             if c.name == "VehicleCollisionTest"), None
        )

        world.tick()
        start_wall = time.time()

        while not adapter.is_done(parking_scenario.car.car.destination):
            world.tick()
            ts = world.get_snapshot().timestamp
            GameTime.on_carla_tick(ts)
            CarlaDataProvider.on_carla_tick()
            parking_scenario.car.car.localize()

            control = adapter.run_step_testbed(ts)
            parking_scenario.car.actor.apply_control(control)

            loc = parking_scenario.car.actor.get_location()
            actual_positions.append((loc.x, loc.y))

            if parking_scenario.scenario_tree:
                parking_scenario.scenario_tree.tick_once()

            tree_done = (parking_scenario.scenario_tree.status
                         != py_trees.common.Status.RUNNING)
            timed_out = time.time() - start_wall > WALL_TIMEOUT
            if tree_done or timed_out:
                reason = "tree stopped" if tree_done else f"wall timeout {WALL_TIMEOUT}s"
                print(f"      [end] {reason}")
                break

        iou        = parking_scenario.car.iou()
        collisions = vehicle_criterion.actual_value if vehicle_criterion else 0
        print(f"      iou={iou:.3f}  collisions={collisions}  ticks={len(actual_positions)}")
        return {"iou": iou, "collisions": collisions,
                "positions": actual_positions, "ticks": len(actual_positions)}

    finally:
        if adapter:
            adapter.destroy_cam()
        if parking_scenario:
            parking_scenario.cleanup()
        world.tick()
        world.tick()


# ---------------------------------------------------------------------------

def check_scenario(result, scenario_idx, dest_spot):
    """Assert the four properties for one scenario result. Returns True if all pass."""
    positions  = result["positions"]
    iou        = result["iou"]
    collisions = result["collisions"]

    print(f"\n  --- Assertions for scenario {scenario_idx} (spot {dest_spot}) ---")
    all_ok = True

    # 1. Aisle phase: no early lateral deviation
    aisle_pos = [(x, y) for x, y in positions if y < AISLE_Y_THRESHOLD]
    if not aisle_pos:
        print(f"    [WARN] No ticks with y < {AISLE_Y_THRESHOLD} — aisle check skipped")
    else:
        lat_devs = [abs(x - EGO_SPAWN_X) for x, y in aisle_pos]
        max_lat  = max(lat_devs)
        print(f"    Aisle phase ({len(aisle_pos)} ticks): max lateral deviation = {max_lat:.2f}m")
        all_ok &= check(
            max_lat < MAX_AISLE_LATERAL_M,
            f"Aisle lateral deviation {max_lat:.2f}m < {MAX_AISLE_LATERAL_M}m",
            "Car went diagonal immediately — guidance still pulling toward dest diagonally",
        )

    # 2. Eastward turn reached
    if positions:
        max_x = max(x for x, y in positions)
        print(f"    Max x reached: {max_x:.2f}m  (threshold {EAST_THRESHOLD}m)")
        all_ok &= check(
            max_x > EAST_THRESHOLD,
            f"Car reached x={max_x:.2f}m > {EAST_THRESHOLD}m (turned East)",
            "Car never turned East — stuck in aisle or went straight South past the spot",
        )
    else:
        all_ok &= fail("No position data recorded")

    # 3. IOU
    all_ok &= check(
        iou >= MIN_IOU,
        f"IOU {iou:.3f} >= {MIN_IOU}  (car parked successfully)",
        f"IOU=0 means complete failure: car never overlapped the parking spot",
    )

    # 4. No collision
    all_ok &= check(
        collisions == 0,
        f"No vehicle collision (got {collisions})",
        "Car collided — route is still steering into parked vehicles",
    )

    return all_ok


# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  CARLA integration test — L-shaped route guidance fix")
    print(f"  Scenarios: {SCENARIOS}")
    print("=" * 65)

    try:
        client = load_client()
        world  = town04_load(client)
        CarlaDataProvider.set_client(client)
        CarlaDataProvider.set_world(world)
        town04_spectator_bev(world)
    except Exception as e:
        print(f"[FATAL] Could not connect to CARLA: {e}")
        sys.exit(1)

    scenario_results = []
    pass_flags       = []

    for i, (dest_spot, parked_spots) in enumerate(SCENARIOS):
        print(f"\n{'='*65}")
        print(f"  Scenario {i+1}/{len(SCENARIOS)}: destination={dest_spot}, parked={parked_spots}")
        print(f"{'='*65}")
        result = run_one_scenario(world, dest_spot, parked_spots)
        passed = check_scenario(result, i + 1, dest_spot)
        scenario_results.append({
            "scenario": i + 1,
            "destination_spot": dest_spot,
            "parked_spots": parked_spots,
            **{k: v for k, v in result.items() if k != "positions"},
            "passed": passed,
        })
        pass_flags.append(passed)

    # ---- aggregate summary ----
    print(f"\n{'='*65}")
    print("  SUMMARY")
    print(f"{'='*65}")
    ious       = [r["iou"]        for r in scenario_results]
    collisions = [r["collisions"] for r in scenario_results]
    print(f"  IOUs:            {[f'{v:.3f}' for v in ious]}  (mean={np.mean(ious):.3f})")
    print(f"  Collisions:      {collisions}")
    print(f"  Scenarios passed: {sum(pass_flags)}/{len(pass_flags)}")

    # Full-failure detection: all IOUs zero and any collision
    if all(v == 0.0 for v in ious):
        print("\n  [FULL FAILURE DETECTED] All IOUs are 0.0 — model never parked successfully.")
    if any(v > 0 for v in collisions):
        print(f"  [COLLISION FAILURE] {sum(collisions)} total collisions across all scenarios.")

    for r in scenario_results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] Scenario {r['scenario']}: spot={r['destination_spot']} "
              f"iou={r['iou']:.3f} coll={r['collisions']}")

    # Write JSON summary next to results dir
    run_dir = os.path.join(_ADAPTER_ROOT, "results",
                           datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_test")
    os.makedirs(run_dir, exist_ok=True)
    summary_path = os.path.join(run_dir, "test_results.json")
    with open(summary_path, "w") as f:
        json.dump(scenario_results, f, indent=2)
    print(f"\n  Results written to: {summary_path}")

    world.tick()
    overall_pass = all(pass_flags)
    print(f"\n  Overall: {'PASSED' if overall_pass else 'FAILED'}")
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
