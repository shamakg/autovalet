import numpy as np

# Mocking Carla classes for testing without a simulator
class Location:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

class Rotation:
    def __init__(self, yaw=0.0):
        self.yaw = yaw

class Transform:
    def __init__(self, location, rotation=Rotation()):
        self.location = location
        self.rotation = rotation
    
    def transform(self, loc):
        # Simplified 2D rotation around Z axis (yaw)
        rad = np.deg2rad(self.rotation.yaw)
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)
        
        nx = loc.x * cos_a - loc.y * sin_a + self.location.x
        ny = loc.x * sin_a + loc.y * cos_a + self.location.y
        return Location(nx, ny, loc.z + self.location.z)

class BoundingBox:
    def __init__(self, extent):
        self.extent = extent

class ActorMock:
    def __init__(self, transform, extent):
        self.transform = transform
        self.bounding_box = BoundingBox(extent)
    def get_transform(self):
        return self.transform

# Re-implementing the distance logic for the unit test
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
    ego_ext = ego_actor.bounding_box.extent
    ego_corners_local = [
        Location(x=-ego_ext.x, y=-ego_ext.y, z=0),
        Location(x=-ego_ext.x, y= ego_ext.y, z=0),
        Location(x= ego_ext.x, y=-ego_ext.y, z=0),
        Location(x= ego_ext.x, y= ego_ext.y, z=0)
    ]
    ego_p = [np.array([ego_trans.transform(c).x, ego_trans.transform(c).y]) for c in ego_corners_local]
    
    door_trans = door_vehicle.get_transform()
    door_ext = door_vehicle.bounding_box.extent
    door_segments = []
    for side in [-1, 1]:
        h_loc = Location(x=0.5*door_ext.x, y=side*door_ext.y, z=0)
        e_loc = Location(x=0.5*door_ext.x, y=side*(door_ext.y + 1.0), z=0)
        h_world = door_trans.transform(h_loc)
        e_world = door_trans.transform(e_loc)
        door_segments.append((np.array([h_world.x, h_world.y]), np.array([e_world.x, e_world.y])))
        
    # Check if any door endpoint is inside ego bounding box
    ego_yaw_rad = np.deg2rad(ego_trans.rotation.yaw)
    cos_a = np.cos(-ego_yaw_rad)
    sin_a = np.sin(-ego_yaw_rad)
    
    min_dist = float('inf')
    for s1, s2 in door_segments:
        # Check intersection by looking at endpoints in ego-local space
        for pt in [s1, s2]:
            dx = pt[0] - ego_trans.location.x
            dy = pt[1] - ego_trans.location.y
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            if abs(lx) <= ego_ext.x and abs(ly) <= ego_ext.y:
                return 0.0

        # Corner to segment
        for p in ego_p:
            min_dist = min(min_dist, distance_point_to_segment(p, s1, s2))
        # Segment endpoints to ego edges
        ego_edges = [(ego_p[0], ego_p[1]), (ego_p[1], ego_p[3]), (ego_p[3], ego_p[2]), (ego_p[2], ego_p[0])]
        for ep1, ep2 in ego_edges:
            min_dist = min(min_dist, distance_point_to_segment(s1, ep1, ep2))
            min_dist = min(min_dist, distance_point_to_segment(s2, ep1, ep2))
    return min_dist

def test_logic():
    # Door vehicle at origin, yaw 0.
    # Extent: x=2.5, y=1.0. 
    # Door hinge at x=1.25, y=+/-1.0. 
    # Door edge at x=1.25, y=+/-2.0.
    door_vehicle = ActorMock(Transform(Location(0,0,0), Rotation(0)), Location(2.5, 1.0, 1.0))
    
    # Ego vehicle with extent: x=2.0, y=1.0.
    
    test_cases = [
        # (Ego X, Ego Y, Expected Dist approx, Expected Safety)
        (1.25, -3.05, 0.05, 0.0),   # Distance 5cm to FR door edge -> Safety 0
        (1.25, -3.30, 0.30, 1.0),   # Distance 30cm to FR door edge -> Safety 1.0
        (1.25, -3.175, 0.175, 0.5), # Distance 17.5cm -> Safety 0.5
        (5.0, 5.0, 2.6575, 1.0),    # Distance ~2.66m -> Safety 1.0
        (1.25, -2.5, 0.0, 0.0),     # Collision with door (segment enters box) -> Safety 0.0
    ]
    
    print("--- Running Mocked Door Safety Metric Tests ---")
    for ex, ey, target_dist, expected_safety in test_cases:
        ego_actor = ActorMock(Transform(Location(ex, ey, 0), Rotation(0)), Location(2.0, 1.0, 1.0))
        dist = calculate_min_distance_to_door(ego_actor, door_vehicle)
        safety_score = np.clip((dist - 0.05) / (0.30 - 0.05), 0.0, 1.0)
        
        print(f"Ego at ({ex}, {ey}):")
        print(f"  Calculated Dist: {dist:.4f}m (Target ~{target_dist:.2f}m)")
        print(f"  Safety Score:    {safety_score:.4f} (Expected ~{expected_safety:.4f})")
        
        # Verify safety score is within tolerance
        assert abs(safety_score - expected_safety) < 1e-3, f"Safety score mismatch! Got {safety_score}, expected {expected_safety}"

    print("\nAll mock tests passed! The distance logic and safety score mapping are correct.")

if __name__ == '__main__':
    test_logic()
