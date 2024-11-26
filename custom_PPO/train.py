import torch
import matplotlib.pyplot as plt
from ibuprofen_env import IbuprofenEnv
from ppo_agent import PPOAgent
from hyperparameter_tuning import run_hyperparameter_tuning

def train_agent(n_episodes=1000, n_trials=20):
    """
    Trains a PPO agent using the best hyperparameters from hyperparameter tuning.

    This function first performs hyperparameter tuning using Optuna to determine
    the optimal PPO parameters, then trains the agent over a specified number of episodes.

    Args:
        n_episodes (int): Number of episodes for training.
        n_trials (int): Number of trials for hyperparameter tuning.

    Returns:
        PPOAgent: The trained PPO agent.
        list: History of total rewards across episodes during training.
    """
    # Run hyperparameter tuning to find the best parameters for PPO
    best_params = run_hyperparameter_tuning(n_trials=n_trials)

    # Initialize the environment and agent
    env = IbuprofenEnv()  # Custom ibuprofen environment
    state_dim = env.observation_space.shape[0]  # State space dimension
    action_dim = env.action_space.n  # Action space dimension

    # Create a PPO agent using the best hyperparameters
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, **best_params)
    reward_history = []  # To store total rewards for each episode

    # Training loop
    for episode in range(n_episodes):
        state = env.reset()  # Reset the environment at the start of each episode
        states, actions, rewards, dones, old_probs = [], [], [], [], []  # Store trajectory data
        total_reward = 0  # Accumulate total reward for the episode

        while True:
            # Convert state to a tensor and compute action probabilities
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_probs = agent.policy(state_tensor).detach().numpy()

            # Sample an action based on the probabilities
            action = torch.multinomial(torch.tensor(action_probs), 1).item()

            # Take the action in the environment
            new_state, reward, done, _ = env.step(action)

            # Record trajectory data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            old_probs.append(action_probs[action])

            # Update state and total reward
            state = new_state
            total_reward += reward

            if done:
                break

        # Train the agent using the collected trajectory
        agent.train(states, actions, rewards, dones, old_probs)

        # Record the total reward for this episode
        reward_history.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    return agent, reward_history

def plot_training(reward_history):
    """
    Plots the training performance of the PPO agent.

    The plot shows the total rewards obtained by the agent for each training episode,
    illustrating the agent's learning progress.

    Args:
        reward_history (list): List of total rewards for each episode.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(reward_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Performance')
    plt.grid()
    plt.show()

if __name__ == "__main__":

    trained_agent, rewards = train_agent(n_episodes=1000, n_trials=20)

    plot_training(rewards)
