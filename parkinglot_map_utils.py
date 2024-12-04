import carla
from leaderboard.autovalet.v2 import CarlaCar
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration
from v2_experiment_utils import load_client
import sys
import os
import time

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarios.pedestrian_parking_v2 import PedestrianCrossingv2
HOST = 'host.docker.internal' ### my Host - SHAMAK
PORT = 2000
DEBUG = True
EGO_VEHICLE = 'vehicle.tesla.model3'

def load_client():
    print(f"starting simulation on {HOST}:{PORT}")
    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)
    return client
def town04_spectator_bev(world):
    spectator_location = carla.Location(x=-36.562950, y=-21.714865, z=20)
    spectator_rotation = carla.Rotation(pitch=-90.0)
    world.get_spectator().set_transform(carla.Transform(spectator_location, spectator_rotation))

def town04_spectator_follow(world, car):
    spectator_rotation = car.actor.get_transform().rotation
    spectator_rotation.pitch -= 20
    spectator_transform = carla.Transform(car.actor.get_transform().transform(carla.Location(x=-10,z=5)), spectator_rotation)
    world.get_spectator().set_transform(spectator_transform)

def town04_load(client):
    world = client.load_world('LargeParkingLotv1')
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    world.apply_settings(settings)
    client.reload_world(False)
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)
    return world

def approximate_bb_from_center(loc, padding=0):
    return [
        loc.x - 2.4 - padding, loc.y - 0.96,
        loc.x + 2.4 + padding, loc.y + 0.96
    ]

def town04_spawn_ego_vehicle(world):
    # destination_parking_spot_loc = parking_vehicle_locations_Town04[destination_parking_spot]
    blueprint = world.get_blueprint_library().filter(EGO_VEHICLE)[0]
    player_location_Town04 = carla.Transform(carla.Location(x=-36.562950, y=-21.714865, z=1))
    destination_parking_spot_loc = carla.Location(x=-50.562950, y=-21.714865, z=1)
    return CarlaCar(world, blueprint, player_location_Town04, destination_parking_spot_loc, approximate_bb_from_center(destination_parking_spot_loc), debug=DEBUG)


def run():
    """Main execution function"""
    try:
        # Connect to CARLA
        client = load_client()
        
        # Load map
        world = town04_load(client)
        
        # Initialize CarlaDataProvider
        CarlaDataProvider.set_client(client)
        CarlaDataProvider.set_world(world)
        world.tick() 
        
        # Set camera view
        town04_spectator_bev(world)
        
        # Spawn ego vehicle
        ego_vehicle = town04_spawn_ego_vehicle(world)
        CarlaDataProvider.register_actor(ego_vehicle)
        
        world.tick() 
        print("Setup complete!")
        print("Press Ctrl+C to exit...")
        
        # Keep simulation running
        try:
            for i in range(50):
                time.sleep(0.1)
                world.tick()
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down...")
            
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        try:
            if ego_vehicle.actor.is_alive:
                ego_vehicle.destroy()
                world.tick()
        except:
            pass
        print("✅ Cleanup complete")

if __name__ == "__main__":
    run()
