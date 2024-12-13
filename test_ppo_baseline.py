import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit, RecordVideo
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter
import os
import pandas as pd

logdir = "logs_baseline"
models_dir = "models_baseline"

# Ambiente com modo de renderização "rgb_array"
env = gym.make("LunarLander-v3", render_mode="rgb_array")

# Envolve o ambiente no RecordVideo para gravar todos os episódios
env = RecordVideo(
    env,
    video_folder="videos_ppo_baseline",
    episode_trigger=lambda episode_id: True,
    name_prefix="video_ppo"
)

# Monitor para registro de métricas
env = Monitor(env, filename=os.path.join(logdir, "monitor.csv"))

model = PPO.load(f"{models_dir}/5")

num_episodes = 5
for episode in range(num_episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    print(f"Episódio {episode+1}: Recompensa Total = {total_reward}")

env.close()
