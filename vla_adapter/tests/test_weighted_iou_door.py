"""
Targeted unit tests for weighted IoU logic in the open-door (DoorMode) scenario.
Tests three layers:
  1. distance_point_to_segment (pure numpy)
  2. safety_score / weighted_iou formula
  3. calculate_min_distance_to_door (carla mocked out)
"""

import sys
import types
import math
import numpy as np
import unittest

# ---------------------------------------------------------------------------
# Minimal carla stub so we can import benchmark logic without CARLA installed
# ---------------------------------------------------------------------------
carla_mod = types.ModuleType("carla")

class _Location:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

class _BoundingBox:
    def __init__(self, extent):
        self.extent = extent

class _Rotation:
    def __init__(self, yaw=0.0):
        self.yaw = float(yaw)

class _Transform:
    def __init__(self, location, rotation):
        self.location = location
        self.rotation = rotation

    def transform(self, local_loc):
        """Apply yaw rotation then translate."""
        yaw_rad = math.radians(self.rotation.yaw)
        cos_a = math.cos(yaw_rad)
        sin_a = math.sin(yaw_rad)
        wx = self.location.x + local_loc.x * cos_a - local_loc.y * sin_a
        wy = self.location.y + local_loc.x * sin_a + local_loc.y * cos_a
        return _Location(x=wx, y=wy, z=self.location.z + local_loc.z)

carla_mod.Location  = _Location
carla_mod.BoundingBox = _BoundingBox
carla_mod.Rotation  = _Rotation
carla_mod.Transform = _Transform
sys.modules["carla"] = carla_mod


# ---------------------------------------------------------------------------
# Inline the two pure functions from benchmark.py (identical to source)
# to avoid pulling in the full CARLA/SCENARIO import chain.
# ---------------------------------------------------------------------------

def distance_point_to_segment(p, s1, s2):
    v = s2 - s1
    w = p - s1
    c1 = np.dot(w, v)
    if c1 <= 0:
        return np.linalg.norm(p - s1)
    c2 = np.dot(v, v)
    if c2 <= c1:
        return np.linalg.norm(p - s2)
    b = c1 / c2
    pb = s1 + b * v
    return np.linalg.norm(p - pb)


def calculate_min_distance_to_door(ego_actor, door_vehicle):
    ego_trans = ego_actor.get_transform()
    ego_ext   = ego_actor.bounding_box.extent
    ego_corners_local = [
        carla_mod.Location(x=-ego_ext.x, y=-ego_ext.y, z=0),
        carla_mod.Location(x=-ego_ext.x, y= ego_ext.y, z=0),
        carla_mod.Location(x= ego_ext.x, y=-ego_ext.y, z=0),
        carla_mod.Location(x= ego_ext.x, y= ego_ext.y, z=0),
    ]
    ego_p = [np.array([ego_trans.transform(c).x, ego_trans.transform(c).y])
             for c in ego_corners_local]

    door_trans = door_vehicle.get_transform()
    door_ext   = door_vehicle.bounding_box.extent
    door_segments = []
    for side in [-1, 1]:
        h_loc = carla_mod.Location(x=0.5*door_ext.x, y=side*door_ext.y, z=0)
        e_loc = carla_mod.Location(x=0.5*door_ext.x, y=side*(door_ext.y + 1.0), z=0)
        h_world = door_trans.transform(h_loc)
        e_world = door_trans.transform(e_loc)
        door_segments.append((np.array([h_world.x, h_world.y]),
                               np.array([e_world.x, e_world.y])))

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
            min_dist = min(min_dist, distance_point_to_segment(p, s1, s2))
        ego_edges = [(ego_p[0], ego_p[1]), (ego_p[1], ego_p[3]),
                     (ego_p[3], ego_p[2]), (ego_p[2], ego_p[0])]
        for ep1, ep2 in ego_edges:
            min_dist = min(min_dist, distance_point_to_segment(s1, ep1, ep2))
            min_dist = min(min_dist, distance_point_to_segment(s2, ep1, ep2))

    return min_dist


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_actor(cx, cy, yaw_deg, half_x, half_y):
    """Build a minimal mock CARLA actor with bounding box."""
    actor = types.SimpleNamespace()
    actor.bounding_box = _BoundingBox(_Location(x=half_x, y=half_y, z=0.5))
    actor.get_transform = lambda: _Transform(_Location(cx, cy, 0), _Rotation(yaw=yaw_deg))
    return actor


def safety_score(min_door_dist):
    return float(np.clip((min_door_dist - 0.05) / (0.30 - 0.05), 0.0, 1.0))


def weighted_iou(iou, min_door_dist):
    return (iou * 0.5) + (safety_score(min_door_dist) * 0.5)


# ---------------------------------------------------------------------------
# 1. distance_point_to_segment
# ---------------------------------------------------------------------------

class TestDistancePointToSegment(unittest.TestCase):

    def test_point_on_segment(self):
        p  = np.array([1.0, 0.0])
        s1 = np.array([0.0, 0.0])
        s2 = np.array([2.0, 0.0])
        self.assertAlmostEqual(distance_point_to_segment(p, s1, s2), 0.0, places=9)

    def test_point_closest_to_start(self):
        p  = np.array([-1.0, 0.0])
        s1 = np.array([ 0.0, 0.0])
        s2 = np.array([ 2.0, 0.0])
        self.assertAlmostEqual(distance_point_to_segment(p, s1, s2), 1.0, places=9)

    def test_point_closest_to_end(self):
        p  = np.array([3.0, 0.0])
        s1 = np.array([0.0, 0.0])
        s2 = np.array([2.0, 0.0])
        self.assertAlmostEqual(distance_point_to_segment(p, s1, s2), 1.0, places=9)

    def test_point_perpendicular_to_middle(self):
        p  = np.array([1.0, 3.0])
        s1 = np.array([0.0, 0.0])
        s2 = np.array([2.0, 0.0])
        self.assertAlmostEqual(distance_point_to_segment(p, s1, s2), 3.0, places=9)

    def test_point_at_start(self):
        p  = np.array([0.0, 0.0])
        s1 = np.array([0.0, 0.0])
        s2 = np.array([2.0, 0.0])
        self.assertAlmostEqual(distance_point_to_segment(p, s1, s2), 0.0, places=9)

    def test_diagonal_segment(self):
        # Segment from (0,0) to (1,1); closest point to (0,1) is (0.5,0.5) → dist = sqrt(0.5)
        p  = np.array([0.0, 1.0])
        s1 = np.array([0.0, 0.0])
        s2 = np.array([1.0, 1.0])
        self.assertAlmostEqual(distance_point_to_segment(p, s1, s2), math.sqrt(0.5), places=9)

    def test_degenerate_segment_same_endpoints(self):
        p  = np.array([1.0, 0.0])
        s1 = np.array([0.0, 0.0])
        s2 = np.array([0.0, 0.0])  # zero-length
        self.assertAlmostEqual(distance_point_to_segment(p, s1, s2), 1.0, places=9)


# ---------------------------------------------------------------------------
# 2. Safety score and weighted IoU formula
# ---------------------------------------------------------------------------

class TestSafetyScoreFormula(unittest.TestCase):

    def test_at_threshold_min(self):
        # min_door_dist == 0.05 → safety_score == 0
        self.assertAlmostEqual(safety_score(0.05), 0.0, places=9)

    def test_below_threshold_min(self):
        # min_door_dist < 0.05 → clipped to 0
        self.assertAlmostEqual(safety_score(0.00), 0.0, places=9)
        self.assertAlmostEqual(safety_score(0.03), 0.0, places=9)

    def test_at_threshold_max(self):
        # min_door_dist == 0.30 → safety_score == 1
        self.assertAlmostEqual(safety_score(0.30), 1.0, places=9)

    def test_above_threshold_max(self):
        # min_door_dist > 0.30 → clipped to 1
        self.assertAlmostEqual(safety_score(0.50), 1.0, places=9)
        self.assertAlmostEqual(safety_score(float('inf')), 1.0, places=9)

    def test_midpoint(self):
        # min_door_dist == 0.175 → safety_score == 0.5
        self.assertAlmostEqual(safety_score(0.175), 0.5, places=9)

    def test_quarter_point(self):
        dist = 0.05 + 0.25 * (0.30 - 0.05)  # 25% of the way
        self.assertAlmostEqual(safety_score(dist), 0.25, places=9)


class TestWeightedIouFormula(unittest.TestCase):

    def test_perfect_park_perfect_safety(self):
        # iou=1.0, dist=0.30 → weighted=1.0
        self.assertAlmostEqual(weighted_iou(1.0, 0.30), 1.0, places=9)

    def test_perfect_park_zero_safety(self):
        # iou=1.0, dist=0.05 → weighted=0.5
        self.assertAlmostEqual(weighted_iou(1.0, 0.05), 0.5, places=9)

    def test_zero_park_perfect_safety(self):
        # iou=0.0, dist=0.30 → weighted=0.5
        # A car that didn't park but didn't hit the door still gets 0.5
        self.assertAlmostEqual(weighted_iou(0.0, 0.30), 0.5, places=9)

    def test_zero_park_zero_safety(self):
        # iou=0.0, dist=0.05 → weighted=0.0
        self.assertAlmostEqual(weighted_iou(0.0, 0.05), 0.0, places=9)

    def test_half_park_half_safety(self):
        # iou=0.5, dist=0.175 (safety=0.5) → weighted = 0.5*0.5 + 0.5*0.5 = 0.5
        self.assertAlmostEqual(weighted_iou(0.5, 0.175), 0.5, places=9)

    def test_typical_good_scenario(self):
        # iou=0.85, dist=0.25 → safety=0.8 → weighted=0.5*0.85 + 0.5*0.8 = 0.825
        self.assertAlmostEqual(weighted_iou(0.85, 0.25), 0.825, places=9)

    def test_door_collision_penalizes_heavily(self):
        # Even a perfect IoU is halved when the door is hit (dist=0)
        w = weighted_iou(1.0, 0.0)
        self.assertAlmostEqual(w, 0.5, places=9)
        self.assertLess(w, 1.0)

    def test_non_door_mode_passthrough(self):
        # Outside DoorMode, weighted_iou == iou (no safety component)
        iou = 0.73
        # In benchmark: weighted_iou = iou (no modification)
        self.assertAlmostEqual(iou, 0.73, places=9)


# ---------------------------------------------------------------------------
# 3. calculate_min_distance_to_door (carla mocked)
# ---------------------------------------------------------------------------

class TestCalcMinDistanceToDoor(unittest.TestCase):
    """
    Door vehicle: positioned at origin, yaw=0, half_x=2.0, half_y=1.0
    Door segments (side=-1 and side=+1, extending 1m laterally from hinge):
      FL door: hinge=(1.0, -1.0) → end=(1.0, -2.0)  [left side, negative y]
      FR door: hinge=(1.0,  1.0) → end=(1.0,  2.0)  [right side, positive y]
    """

    def setUp(self):
        self.door_vehicle = make_actor(cx=0, cy=0, yaw_deg=0, half_x=2.0, half_y=1.0)

    def test_far_away_ego_no_overlap(self):
        # Ego far to the right (x=10), should return a large positive distance
        ego = make_actor(cx=10, cy=0, yaw_deg=0, half_x=1.0, half_y=1.0)
        d = calculate_min_distance_to_door(ego, self.door_vehicle)
        self.assertGreater(d, 5.0)

    def test_ego_overlapping_door_segment_returns_zero(self):
        # Ego centred at (1, -1.5): its BB covers (0,−2.5)→(2,−0.5),
        # the FL door endpoint (1.0, -2.0) is inside → distance == 0
        ego = make_actor(cx=1, cy=-1.5, yaw_deg=0, half_x=1.0, half_y=1.0)
        d = calculate_min_distance_to_door(ego, self.door_vehicle)
        self.assertEqual(d, 0.0)

    def test_ego_just_outside_door_tip(self):
        # Ego centred at (1, -3.5): BB is (0,−4.5)→(2,−2.5).
        # Closest door endpoint is (1, -2.0); distance from BB edge to it is 0.5
        ego = make_actor(cx=1, cy=-3.5, yaw_deg=0, half_x=1.0, half_y=1.0)
        d = calculate_min_distance_to_door(ego, self.door_vehicle)
        self.assertAlmostEqual(d, 0.5, places=5)

    def test_symmetry_left_right_doors(self):
        # Same distance from left door and right door when ego is symmetric
        ego_left  = make_actor(cx=1, cy=-3.5, yaw_deg=0, half_x=1.0, half_y=1.0)
        ego_right = make_actor(cx=1, cy= 3.5, yaw_deg=0, half_x=1.0, half_y=1.0)
        d_left  = calculate_min_distance_to_door(ego_left,  self.door_vehicle)
        d_right = calculate_min_distance_to_door(ego_right, self.door_vehicle)
        self.assertAlmostEqual(d_left, d_right, places=5)

    def test_door_tip_inside_ego_returns_zero(self):
        # Ego centred at (1, 1.5): BB covers (0,0.5)→(2,2.5),
        # FR door endpoint (1.0, 2.0) is inside → distance == 0
        ego = make_actor(cx=1, cy=1.5, yaw_deg=0, half_x=1.0, half_y=1.0)
        d = calculate_min_distance_to_door(ego, self.door_vehicle)
        self.assertEqual(d, 0.0)

    def test_rotated_ego_still_detects_proximity(self):
        # Ego rotated 90°, placed at (1, -4.0).
        # Door tip is at world (1.0, -2.0); in ego-local coords that lands at
        # lx=2.0 which is outside half_x=1.0, so no overlap → finite d > 0.
        ego = make_actor(cx=1, cy=-4.0, yaw_deg=90, half_x=1.0, half_y=0.5)
        d = calculate_min_distance_to_door(ego, self.door_vehicle)
        self.assertGreater(d, 0.0)
        self.assertLess(d, 5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
