import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os
import pandas as pd
from environment import LunarLanderModified

# Diretórios para logs e modelos
logdir = "logs"
models_dir = "models"

# run tensorboard with:
# tensorboard --logdir=logs

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"Created models directory: {models_dir}")

if not os.path.exists(logdir):
    os.makedirs(logdir)
    print(f"Created logs directory: {logdir}")

# Carregar os melhores parâmetros do CSV
trials_df = pd.read_csv("ppo_optimized.csv")
best_trial = trials_df.loc[trials_df['value'].idxmax()]
best_params = {
    "learning_rate": best_trial["params_learning_rate"],
    "gamma": best_trial["params_gamma"],
    "n_steps": int(best_trial["params_n_steps"]),
    "clip_range": best_trial["params_clip_range"],
    "ent_coef": best_trial["params_ent_coef"],
    "vf_coef": best_trial["params_vf_coef"],
    "max_grad_norm": best_trial["params_max_grad_norm"],
    "total_timesteps": int(best_trial["params_total_timesteps"]),
}

# Instanciar e envolver o ambiente
env = LunarLanderModified()
env = Monitor(env, filename=os.path.join(logdir, "monitor.csv"))

# Instanciar o modelo PPO com os melhores parâmetros
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=best_params["learning_rate"],
    n_steps=best_params["n_steps"],
    batch_size=best_params["n_steps"],
    gamma=best_params["gamma"],
    clip_range=best_params["clip_range"],
    ent_coef=best_params["ent_coef"],
    vf_coef=best_params["vf_coef"],
    max_grad_norm=best_params["max_grad_norm"],
    verbose=1,
    tensorboard_log=logdir
)

iters = 0
n_ite = 100000

while iters < n_ite:
    iters += 1
    model.learn(total_timesteps=best_params['total_timesteps'], reset_num_timesteps=False, tb_log_name="PPO_optimized")
    model.save(f"{models_dir}/{best_params['total_timesteps'] * iters}")

# Avaliar o modelo treinado em novos episódios
env = gym.make("LunarLander-v3")
env = Monitor(env, filename=os.path.join(logdir, "monitor.csv"))

model = PPO.load(f"{models_dir}/{best_params['total_timesteps'] * iters}")

num_episodes = 10
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
