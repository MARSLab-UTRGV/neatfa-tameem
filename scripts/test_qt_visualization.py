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

def test_qt_visualization():
    """Test if Qt visualization is working properly."""
    print(f"Testing Qt visualization...")
    print(f"XML file: {XML}")
    
    # Check if we're in a headless environment
    if 'DISPLAY' not in os.environ:
        print("WARNING: No DISPLAY environment variable found. This might be a headless environment.")
        print("Try running: export DISPLAY=:0")
        return
    
    print(f"DISPLAY: {os.environ.get('DISPLAY')}")
    
    # Create environment with visualization enabled
    print("Creating environment with visualization...")
    env = iant_rl.IAntRLEnv(XML, force_no_viz=False)  # Enable visualization
    
    try:
        # Reset and get initial observation
        print("Resetting environment...")
        obs = env.reset()
        print(f"Initial observation: {obs}")
        
        print("Simulation should now be visible in a Qt window.")
        print("You should see:")
        print("1. A 3D visualization window")
        print("2. A robot (footbot) in the arena")
        print("3. The arena with a light source")
        
        # Run a simple action to see movement
        print("Moving robot forward...")
        obs, reward, terminated, truncated, info = env.step(16.0, 16.0, False)
        print(f"Action completed. Reward: {reward}")
        
        # Wait for user to see the visualization
        print("Waiting 10 seconds for you to see the visualization...")
        print("If you don't see a window, there might be a display issue.")
        time.sleep(10)
        
        print("Test completed!")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("Closing environment...")
        env.close()
        print("Test completed!")

if __name__ == "__main__":
    test_qt_visualization()
