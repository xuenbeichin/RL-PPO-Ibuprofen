from stable_baselines3 import PPO

from custom_PPO.ibuprofen_env import IbuprofenEnv
from reward_logging_callback import RewardLoggingCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import optuna
import matplotlib.pyplot as plt

from sb3.hyperparameter_tuning_sb3 import optimize_ppo


def train_agent(env, best_params, total_timesteps=100000):
    """
    Train a PPO agent with the best hyperparameters.

    Args:
        env: The wrapped environment for training.
        best_params: Dictionary of the best hyperparameters.
        total_timesteps: Total timesteps to train the agent.

    Returns:
        final_agent: The trained PPO agent.
        callback: The logging callback with episode rewards.
    """
    callback = RewardLoggingCallback()
    final_agent = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=best_params['learning_rate'],
        batch_size=best_params['batch_size'],
        n_epochs=best_params['n_epochs'],
        gamma=best_params['gamma'],
        gae_lambda=best_params['gae_lambda'],
        clip_range=best_params['clip_range'],
        ent_coef=best_params['ent_coef'],
    )

    final_agent.learn(total_timesteps=total_timesteps, callback=callback)
    return final_agent, callback


def plot_learning_curve(callback):
    """
    Plot the learning curve based on episode rewards logged by the callback.

    Args:
        callback: RewardLoggingCallback used during training.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(callback.episode_rewards)), callback.episode_rewards, label="Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # Initialize the environment
    env = DummyVecEnv([lambda: IbuprofenEnv()])

    # Run Optuna optimization to fetch the best hyperparameters
    study = optuna.create_study(direction="maximize")
    study.optimize(optimize_ppo, n_trials=20)

    # Fetch best hyperparameters
    best_params = study.best_params
    print("Best hyperparameters:", best_params)

    # Train the agent using the best hyperparameters
    final_agent, callback = train_agent(env, best_params, total_timesteps=100000)

    # Plot the learning curve
    plot_learning_curve(callback)
