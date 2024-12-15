import optuna
from environment import LunarLanderModified
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import pandas as pd
import os
from gymnasium.wrappers import TimeLimit
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Definir os diretórios
logdir = "logs"
models_dir = "models"

# Criar diretórios se não existirem
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"Created models directory: {models_dir}")

if not os.path.exists(logdir):
    os.makedirs(logdir)
    print(f"Created logs directory: {logdir}")


# Função objetivo para a otimização bayseana
def objective(trial):
     # Intervalos de valores para cada hiperparâmetro que queremos otimizar
     learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
     gamma = trial.suggest_float("gamma", 0.9, 0.999)
     n_steps = trial.suggest_int("n_steps", 64, 512, log=True)
     clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
     ent_coef = trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True)
     vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
     max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 1.0)
     total_timesteps = trial.suggest_int("total_timesteps", 10_000, 100_000, log=True)

     # Criar o ambiente (modificado)
     env = TimeLimit(LunarLanderModified(), max_episode_steps=100000)
     env = Monitor(env, filename=os.path.join(logdir, "monitor.csv"))

     # Inicializar o modelo PPO
     model = PPO(
         "MlpPolicy",
         env,
         learning_rate=learning_rate,
         n_steps=n_steps,
         batch_size=n_steps,
         gamma=gamma,
         clip_range=clip_range,
         ent_coef=ent_coef,
         vf_coef=vf_coef,
         max_grad_norm=max_grad_norm,
         verbose=0,
     )

     # Treinar o modelo
     model.learn(total_timesteps=total_timesteps)

     # Avaliar o modelo ao longo de 10 episódios
     rewards = []
     for _ in range(10):
         obs, _ = env.reset()
         done = False
         total_reward = 0
         while not done:
             action, _ = model.predict(obs, deterministic=True)
             obs, reward, terminated, truncated, _ = env.step(action)
             done = terminated or truncated
         total_reward += reward
         rewards.append(total_reward)

     # Retornar a média das recompensas ao longo dos 10 episódios
     return sum(rewards) / len(rewards)

# Configurar a otimização, para 500 tentativas
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=500, n_jobs=1, timeout=3600)

# Imprimir os melhores parâmetros encontrados e guardá-los num ficheiro csv
print("Best hyperparameters:", study.best_params)
study.trials_dataframe().to_csv("ppo_optimized.csv")