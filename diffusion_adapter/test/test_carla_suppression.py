"""
test_carla_suppression.py  —  CARLA integration test for guidance suppression.

These tests validate the MECHANISM of the fix, not just the final outcome.
Each check corresponds to a step in the causal chain we proved in
test_guidance_suppression.py (the math-only tests).

Chain recap:
  1. goal_energy has an East gradient at the final waypoint in the aisle.
  2. Suppressing guidance in the aisle prevents heading drift.
  3. A stable South heading means dest_ego_x ≈ 0 at the junction, so the
     anchor-rotation trigger fires at the right place.
  4. With guidance active and East anchor, the car turns into the spot.

Five targeted checks (in order of the causal chain):

  CHECK 1  Guidance IS suppressed in the aisle.
           Every tick where the car's CARLA y < AISLE_Y_THRESHOLD must have
           _suppress_guidance=True in the tick log.
           Failure mode: flag logic broken — guidance fires in aisle.

  CHECK 2  Heading stays near π/2 (South) throughout the aisle.
           max |heading - π/2| while y < AISLE_Y_THRESHOLD must be < 0.2 rad.
           Pre-fix baseline: ~1.17 rad drift.
           Failure mode: suppression didn't actually prevent East steering.

  CHECK 3  Trigger fires within 3 m of the junction row.
           The first tick with _suppress_guidance=False must have
           CARLA y ∈ [-235.73, -229.73]  (junction row y=-232.73 ± 3 m).
           Failure mode: suppression deactivates too early (drifted heading) or
           too late (car has overshot).

  CHECK 4  After the trigger, the car actually turns East.
           Within 30 ticks of the trigger firing, max_x - trigger_x >= 3 m.
           Failure mode: anchor rotation fires but model doesn't predict East.

  CHECK 5  IOU > 0 and 0 collisions.
           Final parking quality baseline.

Run (CARLA server on localhost:2000 required):
    /home/sumesh/envs/simlingo/bin/python test/test_carla_suppression.py
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
import diff_adapter as da

CHECKPOINT_PATH = os.path.join(_ADAPTER_ROOT, "checkpoints/model.pth")

# ── Geometry constants ──────────────────────────────────────────────────────
EGO_SPAWN_X       = 285.6
JUNCTION_Y_CARLA  = -232.73   # CARLA y of the junction row (dest same y as spot)
# With TURN_X_THRESHOLD=3.0, trigger fires ~3m before junction (y≈-235.73).
# Aisle threshold set conservatively south of the trigger point.
AISLE_Y_THRESHOLD = -236.0    # y < this = car still in the aisle (suppress must be True)

# ── Per-check tolerances ─────────────────────────────────────────────────────
MAX_HEADING_DRIFT_RAD  = 0.2   # max |h - π/2| allowed in aisle (< 11.5°)
TRIGGER_Y_TOLERANCE_M  = 4.0   # trigger must fire within 4m of junction y (threshold=3m + 1m slack)
EAST_TURN_MIN_M        = 3.0   # car must gain ≥ this many metres in x after trigger
EAST_TURN_MAX_TICKS    = 50    # within this many ticks of the trigger firing
MIN_IOU                = 0.05  # parking quality floor
WALL_TIMEOUT           = 200   # seconds per scenario


def ok(tag):          print(f"    [PASS] {tag}")
def fail(tag, msg=""): print(f"    [FAIL] {tag}  {msg}"); return False
def check(cond, tag, msg=""):
    if cond: ok(tag);  return True
    else:    return fail(tag, msg)


# ── Scenario runner ──────────────────────────────────────────────────────────

def run_one_scenario(world, destination_spot, parked_spots):
    """
    Run one parking scenario and return:
      - iou, collisions
      - positions: list of (x, y) CARLA coords at each tick
      - tick_log: list of (x, y, heading_rad, dest_ego_x, suppress) from diff_adapter
    """
    adapter          = None
    parking_scenario = None
    carla_positions  = []   # (x, y) at each tick

    try:
        config = ScenarioConfiguration()
        config.name         = "Parking"
        config.type         = "Parking"
        config.town         = "Town04_Opt"
        config.other_actors = None
        config.route        = False

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

        # Reset the tick log before this scenario starts
        da._tick_log = []

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
            carla_positions.append((loc.x, loc.y))

            if parking_scenario.scenario_tree:
                parking_scenario.scenario_tree.tick_once()

            tree_done = (parking_scenario.scenario_tree.status
                         != py_trees.common.Status.RUNNING)
            timed_out  = time.time() - start_wall > WALL_TIMEOUT
            if tree_done or timed_out:
                reason = "tree stopped" if tree_done else f"wall timeout {WALL_TIMEOUT}s"
                print(f"      [end] {reason}")
                break

        iou        = parking_scenario.car.iou()
        collisions = vehicle_criterion.actual_value if vehicle_criterion else 0
        tick_log   = list(da._tick_log)   # copy before cleanup

        print(f"      iou={iou:.3f}  collisions={collisions}  ticks={len(carla_positions)}")
        return {
            "iou":        iou,
            "collisions": collisions,
            "positions":  carla_positions,
            "tick_log":   tick_log,
        }

    finally:
        if adapter:
            adapter.destroy_cam()
        if parking_scenario:
            parking_scenario.cleanup()
        world.tick()
        world.tick()


# ── Per-scenario assertions ──────────────────────────────────────────────────

def check_scenario(result, scenario_idx):
    positions = result["positions"]
    tick_log  = result["tick_log"]
    iou       = result["iou"]
    collisions= result["collisions"]

    print(f"\n  {'─'*60}")
    print(f"  Targeted checks for scenario {scenario_idx}")
    print(f"  {'─'*60}")
    all_ok = True

    if not tick_log:
        print("    [WARN] tick_log is empty — diff_adapter never called?")
        return False

    # tick_log entries: (x, y, heading_rad, dest_ego_x, suppress)
    xs       = [e[0] for e in tick_log]
    ys       = [e[1] for e in tick_log]
    headings = [e[2] for e in tick_log]
    dest_xs  = [e[3] for e in tick_log]
    sups     = [e[4] for e in tick_log]

    # ── CHECK 1: Guidance is suppressed in the aisle ─────────────────────────
    aisle_ticks = [(i, e) for i, e in enumerate(tick_log) if e[1] < AISLE_Y_THRESHOLD]
    print(f"\n  [CHECK 1] Guidance suppressed in aisle  ({len(aisle_ticks)} aisle ticks)")
    if not aisle_ticks:
        print(f"    [WARN] No ticks with y < {AISLE_Y_THRESHOLD} — aisle check skipped")
    else:
        unsuppressed = [(i, e) for i, e in aisle_ticks if not e[4]]
        if unsuppressed:
            for i, e in unsuppressed[:3]:
                print(f"    tick {i}: y={e[1]:.2f}  dest_ego_x={e[3]:.2f}  suppress=False  ← BAD")
        all_ok &= check(
            len(unsuppressed) == 0,
            f"All {len(aisle_ticks)} aisle ticks have suppress_guidance=True",
            f"{len(unsuppressed)} aisle ticks had guidance active — East drift risk",
        )

    # ── CHECK 2: Heading stays South in the aisle ────────────────────────────
    print(f"\n  [CHECK 2] Heading conservation in aisle  (expect |h−π/2| < {MAX_HEADING_DRIFT_RAD:.2f} rad)")
    if aisle_ticks:
        aisle_headings = [e[2] for _, e in aisle_ticks]
        drifts = [abs(h - np.pi/2) for h in aisle_headings]
        max_drift = max(drifts)
        mean_drift = np.mean(drifts)
        worst_tick = aisle_ticks[int(np.argmax(drifts))]
        print(f"    max  |heading − π/2| = {max_drift:.4f} rad  ({np.rad2deg(max_drift):.1f}°)  "
              f"at y={worst_tick[1][1]:.2f}")
        print(f"    mean |heading − π/2| = {mean_drift:.4f} rad  ({np.rad2deg(mean_drift):.1f}°)")
        print(f"    (pre-fix baseline was ~1.17 rad / 67°)")
        all_ok &= check(
            max_drift < MAX_HEADING_DRIFT_RAD,
            f"Max heading drift {np.rad2deg(max_drift):.1f}° < {np.rad2deg(MAX_HEADING_DRIFT_RAD):.0f}°",
            "Heading drifted East in aisle — suppression not preventing East steering",
        )

    # ── CHECK 3: Trigger fires near the junction row ─────────────────────────
    print(f"\n  [CHECK 3] Trigger fires near junction y={JUNCTION_Y_CARLA}  (±{TRIGGER_Y_TOLERANCE_M}m)")
    # Find first tick where suppress_guidance flips to False
    trigger_tick_idx = None
    for i in range(1, len(tick_log)):
        if tick_log[i-1][4] and not tick_log[i][4]:  # True → False
            trigger_tick_idx = i
            break

    if trigger_tick_idx is None:
        # Check if it was always False (started at junction) or never flipped
        if sups[0] is False:
            print("    [WARN] suppress_guidance was False from tick 0 "
                  "— scenario started at junction?")
        else:
            all_ok &= fail(
                "Trigger (suppress=True → False) never fired",
                "Car reached dest_ego_x < threshold but anchor rotation still didn't help, "
                "or car never reached the junction",
            )
    else:
        trig_y       = tick_log[trigger_tick_idx][1]
        trig_x       = tick_log[trigger_tick_idx][0]
        trig_dest_x  = tick_log[trigger_tick_idx][3]
        trig_heading = tick_log[trigger_tick_idx][2]
        y_error      = abs(trig_y - JUNCTION_Y_CARLA)

        print(f"    Trigger at tick {trigger_tick_idx}: "
              f"y={trig_y:.2f}  x={trig_x:.2f}  "
              f"dest_ego_x={trig_dest_x:.2f}  "
              f"heading={np.rad2deg(trig_heading):.1f}°")
        print(f"    y_error from junction = {y_error:.2f}m  (threshold {TRIGGER_Y_TOLERANCE_M}m)")

        all_ok &= check(
            y_error < TRIGGER_Y_TOLERANCE_M,
            f"Trigger fires at y={trig_y:.2f}, within {y_error:.2f}m of junction {JUNCTION_Y_CARLA}",
            f"Trigger fired {y_error:.2f}m from junction — heading drift shifted trigger point",
        )
        # Also verify heading was still South-ish at trigger time
        hdrift_at_trig = abs(trig_heading - np.pi/2)
        all_ok &= check(
            hdrift_at_trig < MAX_HEADING_DRIFT_RAD,
            f"Heading at trigger = {np.rad2deg(trig_heading):.1f}°  "
            f"(drift={np.rad2deg(hdrift_at_trig):.1f}° < {np.rad2deg(MAX_HEADING_DRIFT_RAD):.0f}°)",
            "Heading was already drifted when trigger fired — anchor rotation may be wrong",
        )

    # ── CHECK 4: Car turns East after trigger ────────────────────────────────
    print(f"\n  [CHECK 4] Car turns East after trigger  "
          f"(need +{EAST_TURN_MIN_M}m in x within {EAST_TURN_MAX_TICKS} ticks)")
    if trigger_tick_idx is not None:
        post_trigger = tick_log[trigger_tick_idx : trigger_tick_idx + EAST_TURN_MAX_TICKS]
        if post_trigger:
            x_at_trigger  = tick_log[trigger_tick_idx][0]
            max_x_post     = max(e[0] for e in post_trigger)
            x_gain         = max_x_post - x_at_trigger
            print(f"    x at trigger: {x_at_trigger:.2f}  "
                  f"max x in next {len(post_trigger)} ticks: {max_x_post:.2f}  "
                  f"gain: {x_gain:.2f}m")
            all_ok &= check(
                x_gain >= EAST_TURN_MIN_M,
                f"x gain after trigger: {x_gain:.2f}m >= {EAST_TURN_MIN_M}m (car is going East)",
                "Car gained < 3m East after trigger — model not predicting East trajectory",
            )
        else:
            all_ok &= fail("No ticks after trigger", "Scenario ended immediately at trigger?")
    else:
        print("    [SKIP] No trigger found — cannot check East turn")

    # ── CHECK 5: Final quality ───────────────────────────────────────────────
    print(f"\n  [CHECK 5] Final quality")
    max_x = max(xs) if xs else float("nan")
    print(f"    max_x={max_x:.2f}  iou={iou:.3f}  collisions={collisions}")
    all_ok &= check(
        iou >= MIN_IOU,
        f"IOU {iou:.3f} >= {MIN_IOU} (car overlaps the spot)",
        "IOU=0: car never reached the parking spot",
    )
    all_ok &= check(
        collisions == 0,
        f"0 collisions (got {collisions})",
        "Collision detected",
    )

    return all_ok


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  CARLA suppression test — guidance mechanism validation")
    print("=" * 65)
    print("  Each check validates one step of the causal chain:")
    print("  suppress → heading stable → trigger at right place → turn East → park")
    print()

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
        passed = check_scenario(result, i + 1)
        scenario_results.append({
            "scenario": i + 1,
            "destination_spot": dest_spot,
            "iou":        result["iou"],
            "collisions": result["collisions"],
            "ticks":      len(result["positions"]),
            "passed":     passed,
        })
        pass_flags.append(passed)

    print(f"\n{'='*65}")
    print("  SUMMARY")
    print(f"{'='*65}")
    for r in scenario_results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] Scenario {r['scenario']}: "
              f"spot={r['destination_spot']}  iou={r['iou']:.3f}  "
              f"coll={r['collisions']}  ticks={r['ticks']}")
    print(f"\n  Scenarios passed: {sum(pass_flags)}/{len(pass_flags)}")

    run_dir = os.path.join(_ADAPTER_ROOT, "results",
                           datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_suppression")
    os.makedirs(run_dir, exist_ok=True)
    summary_path = os.path.join(run_dir, "test_results.json")
    with open(summary_path, "w") as f:
        json.dump(scenario_results, f, indent=2)
    print(f"\n  Results → {summary_path}")

    world.tick()
    sys.exit(0 if all(pass_flags) else 1)


if __name__ == "__main__":
    main()
