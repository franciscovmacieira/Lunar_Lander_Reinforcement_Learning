from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter
import os
import pandas as pd
from environment import LunarLanderModified

# Diretórios para logs e modelos
logdir = "logs"
models_dir = "models"
csv_file = "ppo_optimized.csv"

# tensorboard --logdir=logs

# Criar diretórios se não existirem
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# Carregar os melhores parâmetros do CSV
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

# Criar o ambiente com RecordEpisodeStatistics e TimeLimit
env = make_vec_env(lambda: RecordEpisodeStatistics(TimeLimit(LunarLanderModified(), max_episode_steps=1000)), n_envs=1)

# Configurar o TensorBoard
writer = SummaryWriter(logdir)

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
    tensorboard_log=logdir,
)

# Loop de treinamento com registro das recompensas médias por iteração
iters = 0
n_ite = 5

while iters < n_ite:
    iters += 1
    # Treinar o modelo por 'total_timesteps'
    model.learn(total_timesteps=best_params["total_timesteps"], reset_num_timesteps=False, tb_log_name="PPO_optimized")

    # Salvar o modelo
    model.save(f"{models_dir}/{best_params['total_timesteps'] * iters}")

    # Calcular recompensas médias manualmente
    rewards = []
    for _ in range(10):  # Coletar recompensas para 10 episódios
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, info = env.step(action)
            total_reward += reward
            done = terminated
        rewards.append(total_reward)

    # Registrar a recompensa média no TensorBoard
    avg_reward = sum(rewards) / len(rewards)
    writer.add_scalar("Recompensa Média por Iteração", avg_reward, iters)

# Fechar o writer do TensorBoard
writer.close()


