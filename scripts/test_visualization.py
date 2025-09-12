#!/usr/bin/env python3
import sys, os
import time

# Paths
ARGOS_DIR = os.path.join(os.path.dirname(__file__), "..", "argos")
ARGOS_DIR = os.path.abspath(ARGOS_DIR)
# Import env
sys.path.append(os.path.join(ARGOS_DIR, "build"))
import iant_rl

# Use the fixed XML file with absolute paths
XML = os.path.join(ARGOS_DIR, "experiments", "iAnt_rl_fixed.xml")

def test_basic_visualization():
    """Test basic visualization without any trained model."""
    print(f"ARGOS_DIR: {ARGOS_DIR}")
    print(f"XML file: {XML}")
    print(f"XML exists: {os.path.exists(XML)}")
    
    # Create environment with visualization enabled
    print("Creating environment with visualization...")
    env = iant_rl.IAntRLEnv(XML, force_no_viz=False)  # Enable visualization
    
    try:
        # Reset and get initial observation
        print("Resetting environment...")
        obs = env.reset()
        print(f"Initial observation shape: {len(obs)}")
        print(f"Initial observation: {obs}")
        
        # Run a few simple actions to see the robot move
        print("Running simple actions...")
        
        # Forward
        print("Action: Forward")
        obs, reward, terminated, truncated, info = env.step(16.0, 16.0, False)
        print(f"  Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        time.sleep(1.0)  # Wait to see movement
        
        # Turn left
        print("Action: Turn left")
        obs, reward, terminated, truncated, info = env.step(-16.0, 16.0, False)
        print(f"  Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        time.sleep(1.0)
        
        # Forward again
        print("Action: Forward")
        obs, reward, terminated, truncated, info = env.step(16.0, 16.0, False)
        print(f"  Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        time.sleep(1.0)
        
        # Stop
        print("Action: Stop")
        obs, reward, terminated, truncated, info = env.step(0.0, 0.0, False)
        print(f"  Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        time.sleep(2.0)  # Wait longer to see final state
        
        print("Basic visualization test completed!")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("Closing environment...")
        env.close()
        print("Test completed!")

if __name__ == "__main__":
    test_basic_visualization()
