from datetime import datetime
import gymnasium as gym
import optuna

from custom_PPO.PPOAgent import PPOAgent
from custom_PPO.cartpole_env.full_run_cartpole_custom import optimize_ppo
from custom_PPO.cartpole_env.scripts.cartpole_custom_ppo_evaluation import (
    evaluate_agent,
    plot_pole_angles,
    create_eval_env,
)
from custom_PPO.cartpole_env.scripts.cartpole_custom_ppo_training import (
    train_agent,
    plot_rewards,
)


def main(episodes=10000):
    """
    Main function to perform PPO optimization, training, and evaluation
    on the CartPole-v1 environment.

    Args:
        episodes (int): Number of training episodes for the agent. Default is 10,000.
    """
    # Create an Optuna study to optimize hyperparameters
    study = optuna.create_study(direction="maximize")  # Maximize the mean reward
    study.optimize(optimize_ppo, n_trials=100)  # Perform 100 trials for hyperparameter tuning

    # Display the best hyperparameters found
    print("Best Hyperparameters:")
    print(study.best_params)

    # Retrieve the best hyperparameters
    best_params = study.best_params

    # Create an evaluation environment with video recording
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    # Initialize the PPO agent using the optimized hyperparameters
    final_model = PPOAgent(
        state_dim=env.observation_space.shape[0],  # Number of state dimensions
        action_dim=env.action_space.n,            # Number of action dimensions
        lr=best_params["learning_rate"],          # Optimized learning rate
        gamma=best_params["gamma"],               # Discount factor
        eps_clip=best_params["eps_clip"],         # Clipping parameter for PPO
        batch_size=best_params["batch_size"],     # Batch size for training
        ppo_epochs=best_params["ppo_epochs"],     # Number of PPO epochs
        entropy_beta=best_params["entropy_beta"], # Entropy coefficient
        buffer_size=best_params["buffer_size"],   # Size of the buffer
        max_steps=best_params["max_steps"],       # Maximum steps for training
        hidden_units=best_params["hidden_units"], # Number of hidden units in the policy network
        num_layers=best_params["num_layers"],     # Number of layers in the policy network
    )

    # Train the agent with the optimized hyperparameters
    reward_history = train_agent(final_model, env, best_params, episodes)

    # Plot the learning curve (reward history over episodes)
    plot_rewards(reward_history)

    # Setup the environment for evaluation with video recording
    video_folder = "/Users/xuenbei/Desktop/rl_coursework2_02015483/custom_ppo/cartpole_env/videos"
    video_name_prefix = f"cartpole_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    eval_env = create_eval_env(video_folder, video_name_prefix)

    # Evaluate the trained agent in the evaluation environment
    state_trajectory = evaluate_agent(final_model, eval_env, episodes)

    # Plot the pole angles over time during evaluation
    plot_pole_angles(state_trajectory)


if __name__ == "__main__":
    main(episodes=10000)
