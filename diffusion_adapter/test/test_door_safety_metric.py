import os
import sys
import numpy as np
import carla

# Add paths to imports
sys.path.insert(0, '/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet')
sys.path.insert(0, '/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/diffusion_adapter')

from testbed.v2_experiment_utils import load_client, town04_load

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
    # Ego bounding box corners in world coordinates
    ego_trans = ego_actor.get_transform()
    ego_ext = ego_actor.bounding_box.extent
    ego_corners_local = [
        carla.Location(x=-ego_ext.x, y=-ego_ext.y, z=0),
        carla.Location(x=-ego_ext.x, y= ego_ext.y, z=0),
        carla.Location(x= ego_ext.x, y=-ego_ext.y, z=0),
        carla.Location(x= ego_ext.x, y= ego_ext.y, z=0)
    ]
    ego_p = [np.array([ego_trans.transform(c).x, ego_trans.transform(c).y]) for c in ego_corners_local]
    
    # Door segments in world coordinates
    door_trans = door_vehicle.get_transform()
    door_ext = door_vehicle.bounding_box.extent
    door_segments = []
    for side in [-1, 1]:
        h_loc = carla.Location(x=0.5*door_ext.x, y=side*door_ext.y, z=0)
        e_loc = carla.Location(x=0.5*door_ext.x, y=side*(door_ext.y + 1.0), z=0)
        h_world = door_trans.transform(h_loc)
        e_world = door_trans.transform(e_loc)
        door_segments.append((np.array([h_world.x, h_world.y]), np.array([e_world.x, e_world.y])))
        
    min_dist = float('inf')
    for s1, s2 in door_segments:
        for p in ego_p:
            min_dist = min(min_dist, distance_point_to_segment(p, s1, s2))
        ego_edges = [(ego_p[0], ego_p[1]), (ego_p[1], ego_p[3]), (ego_p[3], ego_p[2]), (ego_p[2], ego_p[0])]
        for ep1, ep2 in ego_edges:
            min_dist = min(min_dist, distance_point_to_segment(s1, ep1, ep2))
            min_dist = min(min_dist, distance_point_to_segment(s2, ep1, ep2))
    return min_dist

def run_test():
    client = load_client()
    world = town04_load(client)
    bp_lib = world.get_blueprint_library()
    
    # Spawn parked car
    parked_bp = bp_lib.find('vehicle.audi.a2')
    parked_transform = carla.Transform(carla.Location(x=0, y=0, z=0.5), carla.Rotation(yaw=0))
    parked_actor = world.spawn_actor(parked_bp, parked_transform)
    
    # Spawn ego car
    ego_bp = bp_lib.find('vehicle.audi.etron')
    
    # Define test positions (lateral offset from door edge)
    # Door edge is at y ~ -(ext.y + 1.0) = -(0.9 + 1.0) = -1.9
    # Ego half width ~ 1.1. So ego center at y = -1.9 - 1.1 - offset
    
    test_offsets = [
        (0.02, 0.0),   # 2cm -> Safety 0
        (0.175, 0.5), # 17.5cm (middle of 5-30) -> Safety 0.5
        (0.35, 1.0),  # 35cm -> Safety 1.0
        (1.0, 1.0)    # 100cm -> Safety 1.0
    ]
    
    print("\n--- Door Safety Metric Test ---")
    
    try:
        for offset, expected_safety in test_offsets:
            # Calculate required y position
            # parked_ext_y = parked_actor.bounding_box.extent.y
            # door_edge_y = -(parked_ext_y + 1.0)
            # ego_ext_y = 1.1 # approx
            
            # For simplicity, we'll just move the ego until the calculated min_dist matches our target
            # and then check the safety score logic.
            
            # Start far and move closer
            y_pos = -5.0
            ego_transform = carla.Transform(carla.Location(x=0.5 * parked_actor.bounding_box.extent.x, y=y_pos, z=0.5), carla.Rotation(yaw=0))
            ego_actor = world.spawn_actor(ego_bp, ego_transform)
            
            # Fine-tune y_pos to get the exact offset
            current_dist = calculate_min_distance_to_door(ego_actor, parked_actor)
            # Move ego to achieve target 'offset' as min_dist
            new_y = y_pos + (current_dist - offset)
            ego_actor.set_transform(carla.Transform(carla.Location(x=0.5 * parked_actor.bounding_box.extent.x, y=new_y, z=0.5), carla.Rotation(yaw=0)))
            
            # Re-calculate
            final_dist = calculate_min_distance_to_door(ego_actor, parked_actor)
            safety_score = np.clip((final_dist - 0.05) / (0.30 - 0.05), 0.0, 1.0)
            
            print(f"Test Case: Target Dist={offset:.3f}m")
            print(f"  Calculated Dist: {final_dist:.3f}m")
            print(f"  Safety Score:    {safety_score:.3f} (Expected ~{expected_safety:.3f})")
            
            ego_actor.destroy()
            
    finally:
        parked_actor.destroy()
        print("Test finished and actors cleaned up.")

if __name__ == '__main__':
    run_test()
