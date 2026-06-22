import sys
import os
import time
import json
import torch
import numpy as np
import carla
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Add paths
SCRIPT_DIR = "/home/sumesh/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter"
SIMLINGO_ROOT = os.path.join(SCRIPT_DIR, "simlingo")
sys.path.insert(0, SIMLINGO_ROOT)
sys.path.insert(0, os.path.join(SIMLINGO_ROOT, "team_code"))

from agent_interface import SimLingoAdapter

def capture_and_test():
    print("Step 1: Connecting to CARLA to capture a REAL model input...")
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Initialize the real adapter
    checkpoint_path = os.path.join(SCRIPT_DIR, "model/checkpoints/epoch=013.ckpt/pytorch_model.pt")
    os.environ['SAVE_PATH'] = "/tmp/inference_test/"
    adapter = SimLingoAdapter("localhost", 2000)

    # We need a dummy actor, destination and angle for init_testbed
    # We'll spawn a temporary actor if needed, or use the world spectator
    blueprints = world.get_blueprint_library().filter("vehicle.tesla.model3")
    spawn_points = world.get_map().get_spawn_points()
    actor = world.spawn_actor(blueprints[0], spawn_points[0])

    destination = carla.Location(x=0, y=0, z=0)
    angle = 0.0
    
    adapter.init_testbed(checkpoint_path, world, actor, destination, angle)
    
    # Wait for a frame
    print("Waiting for camera frame...")
    timeout = time.time() + 30
    while adapter.latest_frame is None and time.time() < timeout:
        world.tick()
        time.sleep(0.1)
    
    if adapter.latest_frame is None:
        print("Failed to capture frame!")
        return

    # Mock the input_data assembly (same as run_step_testbed)
    # This captures the EXACT state of a real simulation step
    print("Capturing model input state...")
    
    # We need to run run_step once to let it populate the internal DrivingInput
    timestamp = world.get_snapshot().timestamp
    _ = adapter.run_step_testbed(timestamp)
    
    # Capture the internal DrivingInput that was just prepared
    # We'll need to reach into the agent to get the exact tensor dict
    from simlingo_training.utils.custom_types import DrivingInput
    real_input_dict = adapter.DrivingInput.copy()
    real_input = DrivingInput(**real_input_dict)
    
    print("Real input captured successfully.")
    print("Shutting down adapter...")
    adapter.destroy_cam()
    actor.destroy()
    
    # --- Now we have the REAL input. Let's benchmark it in isolation ---
    print("\nStep 2: Starting Isolated Benchmark (Pure Model Compute)...")
    model = adapter.model
    model.eval()
    device = torch.device('cuda')
    
    # Warmup
    print("Starting Warmup (10 iterations)...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(real_input)
    
    torch.cuda.synchronize()
    
    # Benchmark Loop
    n_iters = 150
    print(f"Starting Benchmark ({n_iters} iterations)...")
    latencies = []

    with torch.no_grad():
        for i in range(n_iters):
            torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model(real_input)

            torch.cuda.synchronize()
            end = time.perf_counter()

            latencies.append((end - start) * 1000.0)
            if (i+1) % 25 == 0:
                print(f"  Completed {i+1}/{n_iters}...")

    # Results
    lats = np.array(latencies)
    print("\n" + "="*40)
    print(" VERIFIED INFERENCE RESULTS (Real Data)")
    print("="*40)
    print(f"Mean:   {np.mean(lats):.2f} ms")
    print(f"Median: {np.median(lats):.2f} ms")
    print(f"P95:    {np.percentile(lats, 95):.2f} ms")
    print(f"P99:    {np.percentile(lats, 99):.2f} ms")
    print("="*40)
    print("This measurement is based on real simulation input data.")

    out_dir = os.path.dirname(os.path.abspath(__file__))

    # Save raw latencies
    json_path = os.path.join(out_dir, 'latency_histogram.json')
    with open(json_path, 'w') as f:
        json.dump({'latencies_ms': latencies}, f, indent=2)
    print(f"Saved raw latencies: {json_path}")

    # Plot histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(latencies, bins=25, color='#3498db', edgecolor='white')
    ax.axvline(np.mean(lats), color='red', linestyle='--', linewidth=1.5,
               label=f'Mean: {np.mean(lats):.1f} ms')
    ax.axvline(np.percentile(lats, 95), color='orange', linestyle='--', linewidth=1.5,
               label=f'P95: {np.percentile(lats, 95):.1f} ms')
    ax.set_xlabel('Inference Latency (ms)')
    ax.set_ylabel('Count')
    ax.set_title('SimLingo VLA Inference Latency (isolated, real frame)')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    png_path = os.path.join(out_dir, 'latency_histogram.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Saved histogram:     {png_path}")

if __name__ == "__main__":
    capture_and_test()
