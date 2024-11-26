import numpy as np
import optuna
import torch
from ibuprofen_env import IbuprofenEnv
from ppo_agent import PPOAgent

def objective(trial):
    """
    Defines the objective function for hyperparameter tuning.

    This function trains a PPO agent in the Ibuprofen environment for a fixed
    number of episodes using hyperparameters suggested by Optuna. It evaluates
    the agent based on its average reward over episodes.

    Args:
        trial (optuna.Trial): An Optuna trial object to suggest hyperparameters.

    Returns:
        float: The mean reward obtained by the agent during training.
    """
    # Initialize the environment and retrieve state/action dimensions
    env = IbuprofenEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Suggest some hyperparameters for PPO
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)  # Learning rate
    gamma = trial.suggest_float('gamma', 0.9, 0.99)  # Discount factor
    eps_clip = trial.suggest_float('eps_clip', 0.1, 0.3)  # Clipping range

    # Create a PPO agent with the suggested hyperparameters
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim,
                     lr=lr, gamma=gamma, eps_clip=eps_clip)

    reward_history = []  # To track rewards for each episode

    # Train the agent over multiple episodes
    for episode in range(50):  # Fixed number of episodes
        state = env.reset()  # Reset environment at the start of each episode
        states, actions, rewards, dones, old_probs = [], [], [], [], []  # Trajectory data
        total_reward = 0

        # Interact with the environment
        while True:
            # Convert state to tensor and get action probabilities
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_probs = agent.policy(state_tensor).detach().numpy()

            # Sample action based on probabilities
            action = np.random.choice(env.action_space.n, p=action_probs)

            # Perform the action in the environment
            new_state, reward, done, _ = env.step(action)

            # Store trajectory information
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            old_probs.append(action_probs[action])

            state = new_state  # Update state
            total_reward += reward

            if done:
                break

        # Train the PPO agent using the collected trajectory
        agent.train(states, actions, rewards, dones, old_probs)

        # Record the total reward for this episode
        reward_history.append(total_reward)

    # Return the mean reward over all episodes
    return np.mean(reward_history)

def run_hyperparameter_tuning(n_trials=20):
    """
    Run hyperparameter tuning for the PPO agent.

    Uses Optuna to optimize hyperparameters for the PPO agent by maximizing
    the average reward obtained in the Ibuprofen environment.

    Args:
        n_trials (int): The number of trials to run for hyperparameter tuning.

    Returns:
        dict: The best hyperparameters found by Optuna.
    """
    # Create an Optuna study to maximize the mean reward
    study = optuna.create_study(direction='maximize')

    # Optimize the objective function over the specified number of trials
    study.optimize(objective, n_trials=n_trials)

    return study.best_params

if __name__ == "__main__":

    best_params = run_hyperparameter_tuning(n_trials=20)

    print("\nBest hyperparameters found:")
    for key, value in best_params.items():
        print(f"{key}: {value}")