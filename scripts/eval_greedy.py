#!/usr/bin/env python3
import os
import sys
import time
import torch
import numpy as np

# Paths
ARGOS_DIR = os.path.join(os.path.dirname(__file__), "..", "argos")
ARGOS_DIR = os.path.abspath(ARGOS_DIR)
XML = os.path.join(ARGOS_DIR, "experiments", "iAnt_rl.xml")

# Import env
sys.path.append(os.path.join(ARGOS_DIR, "build"))
import iant_rl

# Model must match training architecture
from train_dqn import QNetwork, NUM_ACTIONS, OBS_DIM, ACTIONS


def load_model(ckpt_path, device):
    model = QNetwork(OBS_DIM, NUM_ACTIONS).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def select_action(qnet, obs, device):
    with torch.no_grad():
        obs_t = torch.from_numpy(obs).to(device).unsqueeze(0)
        q = qnet(obs_t)
        return int(torch.argmax(q, dim=1).item())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Use last saved checkpoint by default
    ckpt = os.path.join(os.path.dirname(__file__), "runs", "dqn_final.pt")
    if not os.path.exists(ckpt):
        print(f"Checkpoint not found: {ckpt}")
        print("Available checkpoints:")
        runs_dir = os.path.join(os.path.dirname(__file__), "runs")
        if os.path.exists(runs_dir):
            for f in os.listdir(runs_dir):
                if f.endswith('.pt'):
                    print(f"  - {f}")
        return

    print(f"Loading checkpoint: {ckpt}")
    qnet = load_model(ckpt, device)
    print("Model loaded successfully")

    # Enable visualization by passing force_no_viz=False
    print("Creating environment with visualization enabled...")
    env = iant_rl.IAntRLEnv(XML, False)
    print("Environment created")

    # ARGoS libs expect working dir at argos
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    print(f"Changing to: {ARGOS_DIR}")
    os.chdir(ARGOS_DIR)
    try:
        print("Resetting environment...")
        obs = env.reset()
        print(f"Environment reset. Observation shape: {len(obs)}")
        print(f"First few obs values: {obs[:5]}")
        
        steps = 0
        ep_return = 0.0
        print("Starting evaluation loop...")
        while True:
            action = select_action(qnet, np.array(obs, dtype=np.float32), device)
            left, right, lay = ACTIONS[action]
            # print(f"Step {steps}: Action {action} -> left={left}, right={right}, lay={lay}")
            
            obs, reward, terminated, truncated, info = env.step(float(left), float(right), bool(lay))
            ep_return += reward
            steps += 1
            
            print(f"Reward: {reward:.3f}, Total: {ep_return:.3f}, Food Left: {info['food_left']}")
            
            time.sleep(0.02)  # slow down for visualization (~50 FPS)
            if terminated or truncated:
                print(f"Episode done: steps={steps} return={ep_return:.2f} Food Left: {info['food_left']}")
                break
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Closing environment...")
        env.close()
        os.chdir(cwd)
        print("Evaluation complete")


if __name__ == "__main__":
    main()


