#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np

# Paths
ARGOS_DIR = os.path.join(os.path.dirname(__file__), "..", "argos")
ARGOS_DIR = os.path.abspath(ARGOS_DIR)

# Import env
sys.path.append(os.path.join(ARGOS_DIR, "build"))
import iant_rl

# Test PPO components
from train_ppo import ActorCritic, NUM_ACTIONS, OBS_DIM, ACTIONS, PPOBuffer, compute_gae

def test_ppo_components():
    print("Testing PPO components...")
    
    # Test ActorCritic model
    device = torch.device("cpu")
    model = ActorCritic(OBS_DIM, NUM_ACTIONS).to(device)
    print(f"✓ ActorCritic model created with {OBS_DIM} obs dim, {NUM_ACTIONS} actions")
    
    # Test forward pass
    test_obs = torch.randn(1, OBS_DIM)
    action_probs, value = model(test_obs)
    print(f"✓ Forward pass works: action_probs shape {action_probs.shape}, value shape {value.value.shape}")
    
    # Test action selection
    action, log_prob, entropy, value = model.get_action(test_obs)
    print(f"✓ Action selection works: action={action.item()}, log_prob={log_prob.item():.3f}, entropy={entropy.item():.3f}")
    
    # Test buffer
    buffer = PPOBuffer(OBS_DIM, max_size=100)
    print(f"✓ PPOBuffer created with max size 100")
    
    # Test GAE computation
    rewards = np.array([1.0, 0.0, 1.0])
    values = np.array([0.5, 0.3, 0.8])
    dones = np.array([False, False, True])
    advantages, returns = compute_gae(rewards, values, dones)
    print(f"✓ GAE computation works: advantages shape {advantages.shape}, returns shape {returns.shape}")
    
    print("\nAll PPO components working correctly!")
    print(f"Available actions: {len(ACTIONS)}")
    print("\nAction Space Breakdown:")
    print("  Full Speed (16.0):")
    for i, (left, right, lay) in enumerate(ACTIONS[:3]):
        print(f"    Action {i}: left={left:5.1f}, right={right:5.1f}, lay={lay}")
    
    print("  Medium Speed (8.0):")
    for i, (left, right, lay) in enumerate(ACTIONS[3:6]):
        print(f"    Action {i}: left={left:5.1f}, right={right:5.1f}, lay={lay}")
    
    print("  Slow Speed (4.0):")
    for i, (left, right, lay) in enumerate(ACTIONS[6:9]):
        print(f"    Action {i}: left={left:5.1f}, right={right:5.1f}, lay={lay}")
    
    print("  Stop & Mixed:")
    for i, (left, right, lay) in enumerate(ACTIONS[9:]):
        print(f"    Action {i}: left={left:5.1f}, right={right:5.1f}, lay={lay}")
    
    print(f"\nWheel speed range: {min([abs(a[0]) for a in ACTIONS if a[0] != 0]):.1f} to {max([abs(a[0]) for a in ACTIONS if a[0] != 0]):.1f}")
    print("Note: ARGoS foot-bot wheel speeds range from -16 to +16")
    print("This comprehensive action space allows the policy to explore:")
    print("  - Different movement speeds (4, 8, 12, 16)")
    print("  - Various turning behaviors (sharp, medium, gentle)")
    print("  - Complex movements (biased forward, mixed speeds)")
    print("  - Pheromone laying strategies")

if __name__ == "__main__":
    test_ppo_components()
