# main.py
import optuna
from stable_baselines3.common.vec_env import DummyVecEnv

from custom_PPO.ibuprofen_env.scripts.ibuprofen_custom_PPO_evaluation import evaluate_agent, plot_plasma_concentration
from custom_PPO.ibuprofen_env.scripts.ibuprofen_custom_PPO_training import train_agent
from environments.ibuprofen_env import IbuprofenEnv
from sb3.ibuprofen_env_sb3.ibuprofen_full_run_sb3 import optimize_ppo
from sb3.ibuprofen_env_sb3.scripts.train_sb3_ibuprofen import plot_learning_curve


def main(episodes=10000):
    """
    Main function to optimize hyperparameters, train an agent, and evaluate it on the Ibuprofen environment.

    Args:
        episodes (int): The number of training episodes for the PPO agent. Default is 10000.
    """

    # Create an Optuna study to maximize the objective function
    study = optuna.create_study(direction="maximize")
    # Optimize hyperparameters using the defined objective function
    study.optimize(optimize_ppo, n_trials=100)
    # Retrieve the best hyperparameters found during optimization
    best_params = study.best_params
    print("Best Hyperparameters:", best_params)

    # Train the PPO agent using the best hyperparameters
    model, callback = train_agent(best_params, episodes)

    # Plot the learning curve from the training process
    plot_learning_curve(callback.episode_rewards)

    # Setup the environment for evaluation
    # Use a DummyVecEnv wrapper for compatibility with stable-baselines3
    env = DummyVecEnv([lambda: IbuprofenEnv(normalize=True)])
    # Evaluate the trained agent and collect the state trajectory
    state_trajectory = evaluate_agent(env, model)

    # Plot plasma concentration profiles based on the evaluation
    plot_plasma_concentration(state_trajectory)

if __name__ == "__main__":
    main(episodes=10000)
