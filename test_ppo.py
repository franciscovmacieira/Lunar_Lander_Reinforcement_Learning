import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter
import os
import pandas as pd

models_dir = "models"
logdir = "logs"
csv_file = "ppo_optimized.csv"

df = pd.read_csv(csv_file)
best_trial = df.loc[df["value"].idxmax()]
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

# Avaliar o modelo treinado em novos episódios
env = gym.make("LunarLander-v3", render_mode = "human")
env = Monitor(env, filename=os.path.join(logdir, "monitor.csv"))

model = PPO.load(f"{models_dir}/{best_params['total_timesteps'] * 200}")

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