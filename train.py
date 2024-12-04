import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os
from environment import LunarLanderModified
import time

from stable_baselines3.common.env_checker import check_env

# run tensorboard with:
# tensorboard --logdir=logs

models_dir = "models"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"Created models directory: {models_dir}")

if not os.path.exists(logdir):
    os.makedirs(logdir)
    print(f"Created logs directory: {logdir}")

env = LunarLanderModified()

# Wrap the environment with a Monitor wrapper
env = Monitor(env, filename=os.path.join(logdir, "monitor.csv"))

env.reset()

# Set support_multi_env=False to prevent automatic vectorization
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=logdir,)

TIMESTEPS = 10000
iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS * iters}")


