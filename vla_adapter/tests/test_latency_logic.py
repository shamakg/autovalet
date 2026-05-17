import time

class MockControl:
    def __init__(self, value):
        self.value = value

def test_latency_logic():
    latency_ms = 200
    control_queue = []
    current_applied_control = MockControl("BRAKE")
    
    # Simulation ticks (50ms each)
    ticks = [0, 50, 100, 150, 200, 250, 300]
    results = []
    
    print(f"--- Latency Logic Test (Latency: {latency_ms}ms) ---")
    
    for t in ticks:
        # 1. Produce raw control at time t
        raw_control = MockControl(f"CTRL_AT_{t}")
        
        # 2. Add to queue with arrival time
        control_queue.append((t + latency_ms, raw_control))
        
        # 3. Process queue to get control that has "arrived"
        while control_queue and control_queue[0][0] <= t:
            current_applied_control = control_queue.pop(0)[1]
            
        results.append((t, current_applied_control.value))
        print(f"Time: {t:3}ms | Applied: {current_applied_control.value}")

    # Expectations:
    # 0ms: BRAKE (queued 200)
    # 50ms: BRAKE (queued 250)
    # 100ms: BRAKE (queued 300)
    # 150ms: BRAKE (queued 350)
    # 200ms: CTRL_AT_0 (arrived!)
    # 250ms: CTRL_AT_50 (arrived!)
    
    expected = [
        (0, "BRAKE"),
        (50, "BRAKE"),
        (100, "BRAKE"),
        (150, "BRAKE"),
        (200, "CTRL_AT_0"),
        (250, "CTRL_AT_50"),
        (300, "CTRL_AT_100")
    ]
    
    for i, (t, val) in enumerate(results):
        assert val == expected[i][1], f"Mismatch at {t}ms: Got {val}, expected {expected[i][1]}"

    print("\nLatency logic verification passed!")

if __name__ == "__main__":
    test_latency_logic()
