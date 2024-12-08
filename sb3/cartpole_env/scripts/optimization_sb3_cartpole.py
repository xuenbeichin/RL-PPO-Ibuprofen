import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

def optimize_ppo(trial):
    """
    Optimize PPO hyperparameters using Optuna for the CartPole-v1 environment.

    This function sets up the CartPole-v1 environment, defines a PPO model with
    hyperparameters suggested by Optuna, trains the model, and evaluates it to
    return the mean reward over a series of evaluation episodes.

    Args:
        trial (optuna.Trial): An Optuna trial object to suggest hyperparameters.

    Returns:
        float: The mean reward over 50 evaluation episodes, used as the objective value.
    """

    # Create the CartPole environment with RGB rendering for evaluation
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = DummyVecEnv([lambda: env])  # Wrap environment for compatibility with Stable-Baselines3

    # Suggest hyperparameters using Optuna
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)  # Learning rate
    gamma = trial.suggest_float("gamma", 0.90, 0.99)                 # Discount factor
    n_epochs = trial.suggest_int("n_epochs", 3, 10)                  # Number of epochs
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 1e-2, log=True) # Entropy coefficient
    batch_size = trial.suggest_int("batch_size", 32, 512, step=32)   # Batch size
    n_steps = trial.suggest_int("n_steps", 64, 2048, step=64)        # Number of steps per rollout
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)        # GAE lambda
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)         # Clipping range for PPO

    # Initialize and train the PPO model
    model = PPO(
        "MlpPolicy",            # Policy type (Multi-Layer Perceptron)
        env,                    # Environment
        learning_rate=lr,       # Learning rate
        gamma=gamma,            # Discount factor
        n_epochs=n_epochs,      # Number of epochs per training update
        ent_coef=ent_coef,      # Entropy coefficient
        batch_size=batch_size,  # Batch size for training
        n_steps=n_steps,        # Number of steps per environment rollout
        gae_lambda=gae_lambda,  # GAE lambda
        clip_range=clip_range,  # PPO clipping range
        verbose=0,              # Suppress training logs
    )
    model.learn(total_timesteps=10000)  # Train for a fixed number of timesteps

    # Evaluate the trained model
    total_rewards = []  # To store total rewards for each evaluation episode
    for _ in range(50):  # Evaluate over 50 episodes, can change if you want
        state = env.reset()  # Reset the environment
        total_reward = 0  # Initialize total reward for the episode
        done = False  # Flag to track episode completion

        while not done:  # Step through the environment until the episode ends
            action, _ = model.predict(state, deterministic=True)  # Predict action
            state, reward, done, info = env.step(action)          # Take the action
            total_reward += reward  # Accumulate the reward

        total_rewards.append(total_reward)  # Record the total reward for the episode

    # Return the mean reward as the objective value
    return np.mean(total_rewards)