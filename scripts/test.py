import sys, os
# Paths
ARGOS_DIR = os.path.join(os.path.dirname(__file__), "..", "argos")
ARGOS_DIR = os.path.abspath(ARGOS_DIR)
# Import env
sys.path.append(os.path.join(ARGOS_DIR, "build"))
import iant_rl

XML = os.path.join(ARGOS_DIR, "experiments", "iAnt_rl.xml")

class ArgosEnvWrapper:
    """Thin wrapper to handle ARGoS chdir and simplify API."""
    def __init__(self, xml_path):
        self.xml_path = os.path.abspath(xml_path)  # Use absolute path
        self.cwd = None
        self.env = None

    def reset(self):
        # ARGoS resolves libraries relative to argos dir; chdir there
        self.cwd = os.getcwd()
        os.chdir(ARGOS_DIR)
        try:
            if self.env is None:
                self.env = iant_rl.IAntRLEnv(self.xml_path, False)  # Enable visualization
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

# Create environment
env = ArgosEnvWrapper(XML)

# Reset and get initial observation
obs = env.reset()
print(f"Initial observation: {obs}")

while True:
    # Send actions
    obs, reward, terminated, truncated, info = env.step(16.0, 16.0, False)  # Forward
    # print(f"Forward action - obs: {obs}, reward: {reward}")

    obs, reward, terminated, truncated, info = env.step(-16.0, 16.0, False)  # Turn left
    # print(f"Turn left action - obs: {obs}, reward: {reward}")

    if terminated or truncated:
        break

# Close when done
env.close()
print("Simulation completed successfully!")