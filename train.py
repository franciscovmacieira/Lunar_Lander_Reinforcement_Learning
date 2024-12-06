import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os
from environment import LunarLanderModified
import time
import optuna

# Directories for logs and models
logdir = "logs"
models_dir = "models"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"Created models directory: {models_dir}")

if not os.path.exists(logdir):
    os.makedirs(logdir)
    print(f"Created logs directory: {logdir}")

# Objective function for optimization
def objective(trial):
    # Suggest values for hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    n_steps = trial.suggest_int("n_steps", 64, 2048, log=True)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 1.0)
    total_timesteps = trial.suggest_int("total_timesteps", 10_000, 500_000, log=True)

    # Instantiate and wrap the environment
    env = LunarLanderModified()
    env = Monitor(env, filename=os.path.join(logdir, "monitor.csv"))

    # Instantiate the PPO model
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
        verbose=1,
    )

    # Train the model
    model.learn(total_timesteps=total_timesteps)

    # Evaluate the model
    rewards = []
    for _ in range(10):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)

    # Return the average reward
    return sum(rewards) / len(rewards)


# Configure the study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=500, n_jobs=1)

# Display the best hyperparameters
print("Best hyperparameters:", study.best_params)

# Save the results
study.trials_dataframe().to_csv("optuna_trials.csv")

env = LunarLanderModified()
env = Monitor(env, filename=os.path.join(logdir, "monitor.csv"))

env.reset()

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=study.best_params["learning_rate"],
    n_steps=study.best_params["n_steps"],
    batch_size=study.best_params["n_steps"],
    gamma=study.best_params["gamma"],
    clip_range=study.best_params["clip_range"],
    ent_coef=study.best_params["ent_coef"],
    vf_coef=study.best_params["vf_coef"],
    max_grad_norm=study.best_params["max_grad_norm"],
    verbose=1,
)

iters = 0
n_ite = 5000000

while iters < n_ite:
    iters += 1
    model.learn(total_timesteps=study.best_params['total_timesteps'], reset_num_timesteps=False, tb_log_name="PPO_optimized")
    model.save(f"{models_dir}/{study.best_params['total_timesteps'] * iters}")

env = gym.make("LunarLander-v3")
env = Monitor(env, filename=os.path.join(logdir, "monitor.csv"))

env.reset()

model = PPO.load(f"{models_dir}/{study.best_params['total_timesteps'] * iters}")

# Avaliar o agente em alguns episódios
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