from datetime import datetime
import optuna

from custom_PPO.cartpole_env.full_run_cartpole_custom import optimize_ppo
from sb3.cartpole_env.full_run_cartpole_sb3 import plot_learning_curve
from sb3.cartpole_env.scripts.evaluation_sb3_cartpole import (
    create_eval_env,
    evaluate_agent,
    plot_pole_angles,
)
from sb3.cartpole_env.scripts.train_agent_sb3_cartpole import train_and_render_cartpole


def main(episodes=10000):
    """
    Main function to optimize hyperparameters, train an agent, and evaluate it on CartPole-v1.

    Args:
        episodes (int): Number of episodes to train the agent. Default is 10,000.
    """
    # Hyperparameter Optimization using Optuna
    study = optuna.create_study(direction="maximize")  # Maximize the evaluation metric (reward)
    study.optimize(optimize_ppo, n_trials=100)  # Run 100 trials for hyperparameter tuning
    best_params = study.best_params  # Retrieve the best hyperparameters
    print("Best Hyperparameters:", best_params)

    # Train the agent using the best hyperparameters
    final_model, callback = train_and_render_cartpole(best_params, episodes)

    # Plot the learning curve based on training rewards
    plot_learning_curve(callback.episode_rewards)

    # Setup the environment for evaluation with video recording
    video_folder = "/Users/xuenbei/Desktop/rl_coursework2_02015483/sb3/cartpole_env/videos"
    video_name_prefix = f"cartpole_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    eval_env = create_eval_env(video_folder, video_name_prefix)

    #Evaluate the trained agent in the evaluation environment
    state_trajectory = evaluate_agent(final_model, eval_env, episodes)

    # Plot the pole angles over time during evaluation
    plot_pole_angles(state_trajectory)


if __name__ == "__main__":
    main(episodes=10000)
