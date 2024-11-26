import numpy as np
from sb3 import PPO
import optuna

def optimize_ppo(trial, env):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    n_epochs = trial.suggest_int('n_epochs', 3, 10)
    gamma = trial.suggest_float('gamma', 0.9, 0.999)
    gae_lambda = trial.suggest_float('gae_lambda', 0.8, 1.0)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.4)
    ent_coef = trial.suggest_float('ent_coef', 1e-4, 0.01, log=True)

    # Create PPO agent with suggested hyperparameters
    agent = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
    )

    # Train the agent for a fixed number of timesteps
    agent.learn(total_timesteps=5000)

    # Evaluate the agent over multiple episodes
    total_rewards = []
    for _ in range(10):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = agent.predict(state, deterministic=True)
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)

    return np.mean(total_rewards)
