import optuna

from custom_PPO.PPOAgent import PPOAgent
from custom_PPO.cartpole_env.scripts.cartpole_custom_ppo_training import plot_rewards
from custom_PPO.ibuprofen_env.scripts.ibuprofen_custom_PPO_evaluation import evaluate_agent, plot_plasma_concentration
from custom_PPO.ibuprofen_env.scripts.ibuprofen_custom_PPO_training import train_agent
from environments.ibuprofen_env import IbuprofenEnv
from custom_PPO.ibuprofen_env.scripts.ibuprofen_custom_PPO_optimization import optimize_ppo

def main(episodes=10000):
    """
    Main function to run optimization, training, and evaluation.
    Args:
        episodes (int): Number of episodes to run for training. Default is 10000.
    """
    # Run hyperparameter optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(optimize_ppo, n_trials=100)

    # Output the best hyperparameters
    print("Best Hyperparameters:")
    print(study.best_params)

    # Get best parameters
    best_params = study.best_params

    # Initialize environment and agent with best parameters
    env = IbuprofenEnv(normalize=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=best_params["learning_rate"],
        gamma=best_params["gamma"],
        eps_clip=best_params["eps_clip"],
        batch_size=best_params["batch_size"],
        ppo_epochs=best_params["ppo_epochs"],
        entropy_beta=best_params["entropy_beta"],
        buffer_size=best_params["buffer_size"],
        max_steps=best_params["max_steps"],
        hidden_units=best_params["hidden_units"],
        num_layers=best_params["num_layers"],
    )

    # Train the agent for the specified number of episodes
    reward_history = train_agent(agent, env, best_params, episodes=episodes)

    # Plot training rewards
    plot_rewards(reward_history)

    # Evaluate the agent
    plasma_concentration_history = evaluate_agent(agent, env)

    # Plot plasma concentration during evaluation
    plot_plasma_concentration(plasma_concentration_history, env.therapeutic_range)

if __name__ == "__main__":
    main(episodes=10000)
