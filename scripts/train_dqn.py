#!/usr/bin/env python3
import os
import sys
import time
from datetime import datetime
import logging
import math
import random
from collections import deque

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


ARGOS_DIR = os.path.join(os.path.dirname(__file__), "..", "argos")
ARGOS_DIR = os.path.abspath(ARGOS_DIR)
sys.path.append(os.path.join(ARGOS_DIR, "build"))
try:
    import iant_rl  # noqa: E402
except:
    print('Import Error: iant_rl')


logging.basicConfig(
    filename=f'log_{str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))}.txt',  # Specify the log file name
    level=logging.INFO,         # Set the minimum logging level to capture
    format='%(asctime)s - %(levelname)s - %(message)s'  # Define the log message format
)

# Create a logger instance (optional, basicConfig uses the root logger)
logger = logging.getLogger(__name__)


XML_PATH = os.path.join(ARGOS_DIR, "experiments", "iAnt_rl.xml")


class ArgosEnvWrapper:
    """Thin wrapper to handle ARGoS chdir and simplify API."""
    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.cwd = None
        self.env = None

    def reset(self):
        # ARGoS resolves libraries relative to argos dir; chdir there
        self.cwd = os.getcwd()
        os.chdir(ARGOS_DIR)
        try:
            if self.env is None:
                self.env = iant_rl.IAntRLEnv(self.xml_path)
            obs = self.env.reset()
        finally:
            os.chdir(self.cwd)
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        # action: int -> (left, right, lay)
        left, right, lay = ACTIONS[action]
        os.chdir(ARGOS_DIR)
        try:
            obs, reward, terminated, truncated, info = self.env.step(float(left), float(right), bool(lay))
        finally:
            os.chdir(self.cwd)
        return np.array(obs, dtype=np.float32), float(reward), bool(terminated), bool(truncated), info

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


# Discretize action space for simplicity
# Speeds in [-16, 16]. We'll pick gentle speeds to reduce collisions.
SPEED = 16.0
ACTIONS = [
    (SPEED, SPEED, 0),     # 0: forward
    (-SPEED, -SPEED, 0),   # 1: backward
    (SPEED, -SPEED, 0),    # 2: turn right (pivot)
    (-SPEED, SPEED, 0),    # 3: turn left (pivot)
    (0.0, 0.0, 0),         # 4: stop
    (SPEED, SPEED, 1),     # 5: forward + lay pheromone
]
NUM_ACTIONS = len(ACTIONS)
OBS_DIM = 15


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def select_action(qnet, obs, epsilon, device):
    if random.random() < epsilon:
        return random.randrange(NUM_ACTIONS)
    with torch.no_grad():
        obs_t = torch.from_numpy(obs).to(device).unsqueeze(0)
        q = qnet(obs_t)
        return int(torch.argmax(q, dim=1).item())


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ArgosEnvWrapper(XML_PATH)

    qnet = QNetwork(OBS_DIM, NUM_ACTIONS).to(device)
    tgt = QNetwork(OBS_DIM, NUM_ACTIONS).to(device)
    tgt.load_state_dict(qnet.state_dict())
    optimizer = optim.Adam(qnet.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    buffer = deque(maxlen=100_000)
    batch_size = 64
    gamma = 0.99
    epsilon_start, epsilon_end, epsilon_decay = 1.0, 0.1, 20_000
    global_step = 0
    target_update_interval = 1_000

    num_episodes = 200
    max_steps_per_episode = 2_000  # cap even if env limit is higher

    os.makedirs(os.path.join(os.path.dirname(__file__), "runs"), exist_ok=True)

    for ep in range(num_episodes):
        obs = env.reset()
        ep_return = 0.0
        ep_steps = 0
        while True:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-global_step / epsilon_decay)
            action = select_action(qnet, obs, epsilon, device)
            next_obs, reward, terminated, truncated, info = env.step(action)

            buffer.append((obs, action, reward, next_obs, terminated or truncated))
            obs = next_obs
            ep_return += reward
            ep_steps += 1
            global_step += 1

            # Optimize
            if len(buffer) >= batch_size:
                batch = random.sample(buffer, batch_size)
                b_obs = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32, device=device)
                b_act = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
                b_rew = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
                b_next = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32, device=device)
                b_done = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=device)

                q = qnet(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = tgt(b_next).max(1)[0]
                    target = b_rew + gamma * (1.0 - b_done) * next_q
                loss = mse(q, target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(qnet.parameters(), 1.0)
                optimizer.step()

            if global_step % target_update_interval == 0:
                tgt.load_state_dict(qnet.state_dict())

            if terminated or truncated:
                break

        logger.info(f"Episode {ep+1}/{num_episodes} steps={ep_steps} return={ep_return:.2f} food_left={info['food_left']}")

        # Save checkpoint periodically
        if (ep + 1) % 20 == 0:
            ckpt = os.path.join(os.path.dirname(__file__), "runs", f"dqn_ep{ep+1}.pt")
            torch.save({"model": qnet.state_dict()}, ckpt)

    # Final save
    ckpt = os.path.join(os.path.dirname(__file__), "runs", "dqn_final.pt")
    torch.save({"model": qnet.state_dict()}, ckpt)
    env.close()


if __name__ == "__main__":
    train()


