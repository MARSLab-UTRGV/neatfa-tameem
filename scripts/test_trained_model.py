#!/usr/bin/env python3
import sys, os
import torch
import numpy as np
import time

# Paths
ARGOS_DIR = os.path.join(os.path.dirname(__file__), "..", "argos")
ARGOS_DIR = os.path.abspath(ARGOS_DIR)
# Import env
sys.path.append(os.path.join(ARGOS_DIR, "build"))
import iant_rl

# Use the fixed XML file with absolute paths
XML = os.path.join(ARGOS_DIR, "experiments", "iAnt_rl_fixed.xml")

class ArgosEnvWrapper:
    """Thin wrapper to handle ARGoS chdir and simplify API."""
    def __init__(self, xml_path, enable_viz=True):
        self.xml_path = os.path.abspath(xml_path)
        self.cwd = None
        self.env = None
        self.enable_viz = enable_viz

    def reset(self):
        # ARGoS resolves libraries relative to argos dir; chdir there
        self.cwd = os.getcwd()
        os.chdir(ARGOS_DIR)
        try:
            if self.env is None:
                # force_no_viz=False enables visualization
                self.env = iant_rl.IAntRLEnv(self.xml_path, force_no_viz=not self.enable_viz)
            obs = self.env.reset()
        finally:
            os.chdir(self.cwd)
        return obs

    def step(self, left_speed, right_speed, lay_pheromone):
        os.chdir(ARGOS_DIR)
        try:
            obs, reward, terminated, truncated, info = self.env.step(float(left_speed), float(right_speed), bool(lay_pheromone))
        finally:
            os.chdir(self.cwd)
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.env is not None:
            os.chdir(ARGOS_DIR)
            try:
                self.env.close()
            finally:
                os.chdir(self.cwd)
            self.env = None

class QNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)

def load_trained_model(model_path, input_dim=15, output_dim=6):
    """Load a trained DQN model."""
    model = QNetwork(input_dim, output_dim)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def select_action(model, obs, device='cpu'):
    """Select action using the trained model."""
    with torch.no_grad():
        obs_t = torch.from_numpy(obs).to(device).unsqueeze(0)
        q = model(obs_t)
        return int(torch.argmax(q, dim=1).item())

def main():
    # Configuration
    enable_visualization = True  # Set to False for headless mode
    model_path = "runs/dqn_final.pt"  # Path to your trained model
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Available models:")
        runs_dir = "runs"
        if os.path.exists(runs_dir):
            for file in os.listdir(runs_dir):
                if file.endswith('.pt'):
                    print(f"  - {file}")
        return
    
    print(f"Loading trained model from {model_path}")
    print(f"Visualization enabled: {enable_visualization}")
    
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(model_path)
    print(f"Model loaded on device: {device}")
    
    # Create environment with visualization
    env = ArgosEnvWrapper(XML, enable_viz=enable_visualization)
    
    # Action mapping (same as in training)
    SPEED = 16.0
    ACTIONS = [
        (SPEED, SPEED, 0),     # 0: forward
        (-SPEED, -SPEED, 0),   # 1: backward
        (SPEED, -SPEED, 0),    # 2: turn right (pivot)
        (-SPEED, SPEED, 0),    # 3: turn left (pivot)
        (0.0, 0.0, 0),         # 4: stop
        (SPEED, SPEED, 1),     # 5: forward + lay pheromone
    ]
    
    # Run simulation
    print("Starting simulation...")
    obs = env.reset()
    print(f"Initial observation: {obs}")
    
    episode_reward = 0
    step_count = 0
    max_steps = 1000  # Limit steps for testing
    
    try:
        while step_count < max_steps:
            # Select action using trained model
            action = select_action(model, obs, device)
            left_speed, right_speed, lay_pheromone = ACTIONS[action]
            
            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(left_speed, right_speed, lay_pheromone)
            
            episode_reward += reward
            step_count += 1
            
            # Print progress every 50 steps
            if step_count % 50 == 0:
                print(f"Step {step_count}: Action={action} ({left_speed:.1f}, {right_speed:.1f}, {lay_pheromone}), "
                      f"Reward={reward:.3f}, Total={episode_reward:.3f}")
                print(f"Info: {info}")
            
            # Check if episode is done
            if terminated or truncated:
                print(f"Episode finished after {step_count} steps")
                break
            
            # Small delay to make visualization visible
            if enable_visualization:
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    
    finally:
        print(f"Final episode reward: {episode_reward:.3f}")
        print(f"Total steps: {step_count}")
        env.close()
        print("Simulation completed!")

if __name__ == "__main__":
    main()
