from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

# Import the custom ibuprofen environment
from environments.ibuprofen_env import IbuprofenEnv
# Custom callback for logging rewards during training
from sb3.callback_sb3 import RewardLoggingCallback


def train_agent(best_params, episodes=10000):
    """
    Trains a PPO agent using the given hyperparameters and episodes.

    Args:
        best_params (dict): A dictionary of the best hyperparameters obtained from optimization.
            Expected keys: "learning_rate", "gamma", "n_epochs", "ent_coef",
            "batch_size", "n_steps", "gae_lambda", "clip_range".
        episodes (int): Number of episodes to train the agent. Default is 10,000.

    Returns:
        tuple:
            - model (PPO): The trained PPO model.
            - callback (RewardLoggingCallback): Callback containing logged rewards.
    """
    # Setup the training environment
    env = DummyVecEnv([lambda: IbuprofenEnv(normalize=True)])  # Normalize the environment for stable learning

    # Initialize the PPO model with the best hyperparameters
    model = PPO(
        "MlpPolicy",  # Use a Multi-Layer Perceptron (MLP) policy
        env,  # Training environment
        learning_rate=best_params["learning_rate"],  # Optimized learning rate
        gamma=best_params["gamma"],  # Optimized discount factor
        n_epochs=best_params["n_epochs"],  # Optimized number of epochs
        ent_coef=best_params["ent_coef"],  # Optimized entropy coefficient
        batch_size=best_params["batch_size"],  # Optimized batch size
        n_steps=best_params["n_steps"],  # Optimized number of steps per update
        gae_lambda=best_params["gae_lambda"],  # Optimized GAE lambda
        clip_range=best_params["clip_range"],  # Optimized clipping range for PPO
        verbose=1,  # Verbose level for training logs
    )

    # Initialize a custom callback for logging rewards
    callback = RewardLoggingCallback()

    # Train the PPO model for the specified number of timesteps
    model.learn(total_timesteps=episodes * best_params["n_steps"], callback=callback)

    return model, callback


def plot_learning_curve(rewards):
    """
    Plots the learning curve showing rewards per episode during training.

    Args:
        rewards (list or array-like): A list of rewards logged during training.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Learning Curve for Ibuprofen Env (SB3)")
    plt.grid()
    plt.show()
