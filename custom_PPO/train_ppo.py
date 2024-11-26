import numpy as np
import torch
import matplotlib.pyplot as plt
from custom_PPO.full_run import IbuprofenEnv, PPOAgent

def train_agent(agent, env, episodes=1000):
    """
    Train the PPO agent using the provided environment for a specified number of episodes.

    The training loop involves the agent interacting with the environment, collecting experiences, and updating
    its policy and value networks using the PPO algorithm. The total reward per episode is tracked and returned.

    Args:
        agent (PPOAgent): The PPO agent to train.
        env (IbuprofenEnv): The environment in which the agent interacts.
        episodes (int, optional): The number of episodes to train the agent (default is 1000).

    Returns:
        list: A list containing the total reward for each episode.
    """
    reward_history = []

    for episode in range(episodes):
        state = env.reset()  # Reset environment for each episode
        states, actions, rewards, dones, old_probs = [], [], [], [], []  # Initialize buffers for states, actions, rewards, etc.
        total_reward = 0  # Track total reward for the episode

        while True:
            # Select action based on policy
            state_tensor = torch.tensor(state, dtype=torch.float32)  # Convert state to tensor
            action_probs = agent.policy(state_tensor).detach().numpy()  # Get action probabilities from the policy network
            action = np.random.choice(env.action_space.n, p=action_probs)  # Sample an action based on the probabilities

            # Take action in the environment
            new_state, reward, done, _ = env.step(action)  # Perform the action and get new state, reward, and done flag

            # Store transition data for training
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            old_probs.append(action_probs[action])  # Store the action probability for this action

            state = new_state  # Update the state
            total_reward += reward  # Accumulate reward for the episode
            if done:  # End episode if done
                break

        # Train the agent using the collected data
        agent.train(states, actions, rewards, dones, old_probs)
        reward_history.append(total_reward)  # Append the total reward for this episode to the history
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    return reward_history  # Return the reward history for all episodes

def plot_training(reward_history):
    """
    Plot the training performance of the PPO agent over episodes.

    This function takes the reward history from training and visualizes it to observe the agent's learning performance
    over time. A line plot is generated showing the total reward for each episode.

    Args:
        reward_history (list): A list of total rewards per episode during training.
    """
    plt.figure(figsize=(12, 6))  # Set figure size for the plot
    plt.plot(reward_history)  # Plot total rewards for each episode
    plt.xlabel('Episode')  # Label for the x-axis
    plt.ylabel('Total Reward')  # Label for the y-axis
    plt.title('PPO Training Performance for Ibuprofen Delivery')  # Title of the plot
    plt.grid()  # Add a grid for easier reading of the plot
    plt.show()  # Display the plot

if __name__ == "__main__":

    # Initialize environment and agent
    env = IbuprofenEnv()  # Create the Ibuprofen environment
    state_dim = env.observation_space.shape[0]  # Get the state dimension from the environment
    action_dim = env.action_space.n  # Get the action dimension (number of actions)
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)  # Initialize the PPO agent

    # Train the agent
    reward_history = train_agent(agent, env, episodes=1000)  # Train the agent for 1000 episodes

    # Plot training performance
    plot_training(reward_history)  # Plot the training rewards over episodes
