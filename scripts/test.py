import os, sys

# Point to argos/build so Python can import the module
ARGOS_DIR = "/home/tameem/Tameem-Work/neatfa-tameem/argos"
sys.path.append(os.path.join(ARGOS_DIR, "build"))

import iant_rl

XML = os.path.join(ARGOS_DIR, "experiments", "iAnt_rl.xml")

# Temporarily chdir so ARGoS loads libs from build/source/...
cwd = os.getcwd()
os.chdir(ARGOS_DIR)
try:
    env = iant_rl.IAntRLEnv(XML)
    obs = env.reset()
finally:
    os.chdir(cwd)

obs, reward, terminated, truncated, info = env.step(0.0, 0.0, False)
print(obs)
env.close()