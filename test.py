import carla
from v2_experiment_utils import load_client

# Connect to CARLA
client = load_client()
client.set_timeout(30.0)


# load map
world = client.load_world('LargeParkingLotv1')
carla_map = world.get_map()

# Default spawn point
default = carla_map.get_spawn_points()[0]

# Set spectator above default spawn
spectator_location = carla.Location(
    x=default.location.x,
    y=default.location.y,
    z= 20  # increase this number to go higher
)

spectator_rotation = carla.Rotation(pitch=-90.0)
world.get_spectator().set_transform(carla.Transform(spectator_location, spectator_rotation))

# Generate waypoints
waypoints = carla_map.generate_waypoints(1.0)
print(f"Number of waypoints: {len(waypoints)}")

# Plot waypoints
xs = [wp.transform.location.x for wp in waypoints]
ys = [wp.transform.location.y for wp in waypoints]

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.scatter(xs, ys, s=1, c='blue')
plt.title("Waypoint map of LargeParkingLotv1")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis('equal')
plt.savefig('waypoints.png')
