import numpy as np
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from environments.ibuprofen_env import IbuprofenEnv


def optimize_ppo(trial):
    """
    Objective function for optimizing PPO hyperparameters using Optuna.

    This function sets up a DummyVecEnv environment with the custom IbuprofenEnv,
    defines a PPO model with hyperparameters suggested by the Optuna trial, trains
    the model, and evaluates its performance. The mean reward from multiple evaluation
    episodes is returned as the optimization objective.

    Args:
        trial (optuna.Trial): A single Optuna trial object to suggest hyperparameters.

    Returns:
        float: The mean reward obtained from the evaluation episodes.
    """
    # Create a vectorized environment using the custom IbuprofenEnv
    env = DummyVecEnv([lambda: IbuprofenEnv(normalize=True)])

    # Define hyperparameters to optimize for Optuna
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)  # Learning rate
    gamma = trial.suggest_float("gamma", 0.90, 0.99)  # Discount factor
    n_epochs = trial.suggest_int("n_epochs", 3, 10)  # Number of epochs
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 1e-2, log=True)  # Entropy coefficient
    batch_size = trial.suggest_int("batch_size", 32, 512, step=32)  # Batch size
    n_steps = trial.suggest_int("n_steps", 64, 2048, step=64)  # Number of steps per update
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)  # GAE (Generalized Advantage Estimation) lambda
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)  # Clipping range for PPO

    # Create a PPO model with the suggested hyperparameters
    model = PPO(
        "MlpPolicy",  # Use a Multi-Layer Perceptron (MLP) policy
        env,  # Environment
        learning_rate=lr,
        gamma=gamma,
        n_epochs=n_epochs,
        ent_coef=ent_coef,
        batch_size=batch_size,
        n_steps=n_steps,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        verbose=0,  # Suppress training output
    )

    # Train the model for a total of 10,000 timesteps
    model.learn(total_timesteps=10000)

    # Evaluate the trained model
    rewards = []  # List to store rewards from evaluation episodes
    for _ in range(50):  # Perform 50 evaluation episodes, can change if you want
        obs = env.reset()  # Reset the environment
        total_reward = 0  # Initialize total reward for this episode
        done = False  # Initialize the done flag
        while not done:  # Loop until the episode ends
            action, _ = model.predict(obs, deterministic=True)  # Get action from the model
            obs, reward, done, _ = env.step(action)  # Take the action in the environment
            total_reward += reward  # Accumulate the reward
        rewards.append(total_reward)  # Store the total reward for this episode

    # Return the mean reward across all evaluation episodes as the objective value
    return np.mean(rewards)
