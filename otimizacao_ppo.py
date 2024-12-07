import optuna

# Função objetivo para otimização
def objective(trial):
     # Sugestões de valores para hiperparâmetros
     learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
     gamma = trial.suggest_float("gamma", 0.9, 0.999)
     n_steps = trial.suggest_int("n_steps", 64, 2048, log=True)
     clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
     ent_coef = trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True)
     vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
     max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 1.0)
     total_timesteps = trial.suggest_int("total_timesteps", 10_000, 500_000, log=True)

     # Instanciar e envolver o ambiente
     env = LunarLanderModified()
     env = Monitor(env, filename=os.path.join(logdir, "monitor.csv"))

     # Instanciar o modelo PPO
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

     # Treinar o modelo
     model.learn(total_timesteps=total_timesteps)

     # Avaliar o modelo
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

     # Retorna a média das recompensas
     return sum(rewards) / len(rewards)

# Configurar o estudo
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=500, n_jobs=1)

# Exibir os melhores hiperparâmetros
print("Best hyperparameters:", study.best_params)

#Salvar os resultados
study.trials_dataframe().to_csv("ppo_optimized.csv")