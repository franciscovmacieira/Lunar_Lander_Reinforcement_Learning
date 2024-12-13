import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter
import os
import pandas as pd
from environment import LunarLanderModified

# Diretórios para logs e modelos
logdir = "logs_baseline"
models_dir = "models_baseline"

# tensorboard --logdir=logs_baseline

# Criar diretórios se não existirem
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


# Criar o ambiente com RecordEpisodeStatistics e TimeLimit
env = gym.make("LunarLander-v3", render_mode="rgb_array")

# Configurar o TensorBoard
writer = SummaryWriter(logdir)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=logdir,)

# Loop de treinamento com registro das recompensas médias por iteração
iters = 0
n_ite = 5

while iters < n_ite:
    iters += 1
    model.learn(total_timesteps = 10000, reset_num_timesteps=False, tb_log_name="PPO_baseline")

    # Salvar o modelo
    model.save(f"{models_dir}/{iters}")

    # Calcular recompensas médias manualmente
    rewards = []
    for _ in range(10):  # Coletar recompensas para 10 episódios
        obs, info = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated
        rewards.append(total_reward)

    # Registrar a recompensa média no TensorBoard
    avg_reward = sum(rewards) / len(rewards)
    writer.add_scalar("Recompensa Média por Iteração", avg_reward, iters)

# Fechar o writer do TensorBoard
writer.close()