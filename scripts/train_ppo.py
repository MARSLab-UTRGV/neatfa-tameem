#!/usr/bin/env python3
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import logging
from datetime import datetime

# Paths
ARGOS_DIR = os.path.join(os.path.dirname(__file__), "..", "argos")
ARGOS_DIR = os.path.abspath(ARGOS_DIR)
XML = os.path.join(ARGOS_DIR, "experiments", "iAnt_rl.xml")

# Import env
sys.path.append(os.path.join(ARGOS_DIR, "build"))
import iant_rl

# Constants
OBS_DIM = 15
NUM_ACTIONS = 15  # Increased from 9 to 15 for more speed variety
ACTIONS = [
    # Full speed movements
    (16.0, 16.0, False),   # 0: Full speed forward
    (16.0, -16.0, False),  # 1: Full speed turn right
    (-16.0, 16.0, False),  # 2: Full speed turn left
    
    # Medium speed movements
    (8.0, 8.0, False),     # 3: Medium speed forward
    (8.0, -8.0, False),    # 4: Medium speed turn right
    (-8.0, 8.0, False),    # 5: Medium speed turn left
    
    # Slow speed movements
    (4.0, 4.0, False),     # 6: Slow forward
    (4.0, -4.0, False),    # 7: Slow turn right
    (-4.0, 4.0, False),    # 8: Slow turn left
    
    # Stop and pheromone actions
    (0.0, 0.0, False),     # 9: Stop
    (0.0, 0.0, True),      # 10: Stop + pheromone
    
    # Mixed speeds for complex movements
    (12.0, 8.0, False),    # 11: Forward with slight right bias
    (8.0, 12.0, False),    # 12: Forward with slight left bias
    (12.0, 8.0, True),     # 13: Forward with slight right bias + pheromone
    (8.0, 12.0, True),     # 14: Forward with slight left bias + pheromone
]

# Hyperparameters
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPSILON = 0.2
PPO_EPOCHS = 4
BATCH_SIZE = 64
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, num_actions):
        super(ActorCritic, self).__init__()
        
        # Shared feature layers
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, obs):
        features = self.feature_net(obs)
        action_probs = F.softmax(self.actor(features), dim=-1)
        value = self.critic(features)
        return action_probs, value
    
    def get_action(self, obs, action=None):
        action_probs, value = self.forward(obs)
        dist = Categorical(action_probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value


class PPOBuffer:
    def __init__(self, obs_dim, max_size=10000):
        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros(max_size, dtype=np.int64)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.values = np.zeros(max_size, dtype=np.float32)
        self.log_probs = np.zeros(max_size, dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=bool)
        self.ptr = 0
        self.size = 0
        self.max_size = max_size
    
    def add(self, obs, action, reward, value, log_prob, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def get_batches(self, batch_size):
        indices = np.random.permutation(self.size)
        for i in range(0, self.size, batch_size):
            batch_indices = indices[i:i + batch_size]
            yield (
                self.obs[batch_indices],
                self.actions[batch_indices],
                self.rewards[batch_indices],
                self.values[batch_indices],
                self.log_probs[batch_indices],
                self.dones[batch_indices]
            )
    
    def clear(self):
        self.ptr = 0
        self.size = 0


def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """Compute Generalized Advantage Estimation"""
    advantages = np.zeros_like(rewards)
    last_advantage = 0
    last_value = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = delta + gamma * gae_lambda * (1 - dones[t]) * last_advantage
        last_advantage = advantages[t]
    
    returns = advantages + values
    return advantages, returns


def train_ppo(model, optimizer, buffer, device):
    """Train PPO for one epoch"""
    model.train()
    
    total_loss = 0
    total_value_loss = 0
    total_policy_loss = 0
    total_entropy_loss = 0
    
    for batch in buffer.get_batches(BATCH_SIZE):
        obs, actions, rewards, old_values, old_log_probs, dones = batch
        
        # Convert to tensors
        obs = torch.FloatTensor(obs).to(device)
        actions = torch.LongTensor(actions).to(device)
        old_values = torch.FloatTensor(old_values).to(device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(device)
        
        # Compute GAE
        advantages, returns = compute_gae(rewards, old_values.cpu().numpy(), dones, GAMMA, GAE_LAMBDA)
        advantages = torch.FloatTensor(advantages).to(device)
        returns = torch.FloatTensor(returns).to(device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Forward pass
        action_probs, values = model(obs)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        # Compute ratios
        ratios = torch.exp(log_probs - old_log_probs)
        
        # PPO clipped objective
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - PPO_EPSILON, 1 + PPO_EPSILON) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # Entropy bonus for exploration
        entropy_loss = -entropy.mean()
        
        # Total loss
        loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        
        total_loss += loss.item()
        total_value_loss += value_loss.item()
        total_policy_loss += policy_loss.item()
        total_entropy_loss += entropy_loss.item()
    
    num_batches = max(1, buffer.size // BATCH_SIZE)
    return {
        'total_loss': total_loss / num_batches,
        'value_loss': total_value_loss / num_batches,
        'policy_loss': total_policy_loss / num_batches,
        'entropy_loss': total_entropy_loss / num_batches
    }


def main():
    # Setup logging
    log_dir = os.path.join(os.path.dirname(__file__), "runs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    log_file = os.path.join(log_dir, f"ppo_log_{timestamp}.txt")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Create model and optimizer
    model = ActorCritic(OBS_DIM, NUM_ACTIONS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Create environment
    env = iant_rl.IAntRLEnv(XML, force_no_viz=True)
    
    # Training parameters
    num_episodes = 200
    max_steps_per_episode = 1000
    update_frequency = 5  # Update every N episodes
    
    # Buffer for collecting experience
    buffer = PPOBuffer(OBS_DIM, max_size=10000)
    
    # ARGoS libs expect working dir at argos
    cwd = os.getcwd()
    os.chdir(ARGOS_DIR)
    
    try:
        episode_returns = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            obs = env.reset()
            episode_return = 0
            episode_length = 0
            
            # Collect experience for this episode
            episode_obs = []
            episode_actions = []
            episode_rewards = []
            episode_values = []
            episode_log_probs = []
            episode_dones = []
            info = None
            
            while True:
                # Convert observation to tensor
                obs_tensor = torch.FloatTensor(obs).to(device).unsqueeze(0)
                
                # Get action from model
                with torch.no_grad():
                    action, log_prob, entropy, value = model.get_action(obs_tensor)
                    action = action.item()
                    value = value.item()
                    log_prob = log_prob.item()
                
                # Take action
                left, right, lay = ACTIONS[action]
                next_obs, reward, terminated, truncated, info = env.step(
                    float(left), float(right), bool(lay)
                )
                
                done = terminated or truncated
                info = info
                
                # Store transition
                episode_obs.append(obs)
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_values.append(value)
                episode_log_probs.append(log_prob)
                episode_dones.append(done)
                
                obs = next_obs
                episode_return += reward
                episode_length += 1
                
                if done:
                    break
            
            # Add episode to buffer
            for i in range(len(episode_obs)):
                buffer.add(
                    episode_obs[i],
                    episode_actions[i],
                    episode_rewards[i],
                    episode_values[i],
                    episode_log_probs[i],
                    episode_dones[i]
                )
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            
            # Log episode info
            logging.info(f"Episode {episode + 1}/{num_episodes} "
                        f"steps={episode_length} return={episode_return:.2f} "
                        f"food_left={info['food_left']}")
            
            # Update policy every N episodes
            if (episode + 1) % update_frequency == 0 and buffer.size >= BATCH_SIZE:
                logging.info(f"Updating policy at episode {episode + 1}")
                
                # Train for multiple epochs
                for epoch in range(PPO_EPOCHS):
                    train_info = train_ppo(model, optimizer, buffer, device)
                    logging.info(f"  Epoch {epoch + 1}/{PPO_EPOCHS}: "
                               f"loss={train_info['total_loss']:.4f}, "
                               f"policy_loss={train_info['policy_loss']:.4f}, "
                               f"value_loss={train_info['value_loss']:.4f}")
                
                # Clear buffer after update
                buffer.clear()
                
                # Save checkpoint
                checkpoint_path = os.path.join(log_dir, f"ppo_episode_{episode + 1}.pt")
                torch.save({
                    'episode': episode + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'episode_returns': episode_returns,
                    'episode_lengths': episode_lengths,
                    'hyperparameters': {
                        'learning_rate': LEARNING_RATE,
                        'gamma': GAMMA,
                        'gae_lambda': GAE_LAMBDA,
                        'ppo_epsilon': PPO_EPSILON,
                        'ppo_epochs': PPO_EPOCHS,
                        'batch_size': BATCH_SIZE
                    }
                }, checkpoint_path)
                logging.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save final model
        final_checkpoint = os.path.join(log_dir, "ppo_final.pt")
        torch.save({
            'episode': num_episodes,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'episode_returns': episode_returns,
            'episode_lengths': episode_lengths,
            'hyperparameters': {
                'learning_rate': LEARNING_RATE,
                'gamma': GAMMA,
                'gae_lambda': GAE_LAMBDA,
                'ppo_epsilon': PPO_EPSILON,
                'ppo_epochs': PPO_EPOCHS,
                'batch_size': BATCH_SIZE
            }
        }, final_checkpoint)
        logging.info(f"Training complete. Final model saved: {final_checkpoint}")
        
        # Print final statistics
        avg_return = np.mean(episode_returns[-50:]) if len(episode_returns) >= 50 else np.mean(episode_returns)
        avg_length = np.mean(episode_lengths[-50:]) if len(episode_lengths) >= 50 else np.mean(episode_lengths)
        logging.info(f"Final 50 episodes - Avg return: {avg_return:.2f}, Avg length: {avg_length:.1f}")
        
    except Exception as e:
        logging.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        os.chdir(cwd)


if __name__ == "__main__":
    main()
