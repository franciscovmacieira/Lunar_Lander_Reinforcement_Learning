import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit, RecordVideo
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter
import os
import pandas as pd

# Definir os diretórios
logdir = "logs_baseline"
models_dir = "models_baseline"

# Carregar o ambiente original do gymnasium
env = gym.make("LunarLander-v3", render_mode="rgb_array")

# Preparar o ambiente para ser capaz de registar cada episódio em vídeo
env = RecordVideo(
    env,
    video_folder="videos_ppo_baseline",
    episode_trigger=lambda episode_id: True,
    name_prefix="video_ppo"
)

# Monitor para o registo das métricas
env = Monitor(env, filename=os.path.join(logdir, "monitor.csv"))

# Carregar o modelo treinado após as 5 iterações
model = PPO.load(f"{models_dir}/5")

# Testar o modelo ao longo de 5 episódios e imprimir as recompensas
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
