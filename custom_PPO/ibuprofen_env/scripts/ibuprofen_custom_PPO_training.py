import torch
import numpy as np
from matplotlib import pyplot as plt

def train_agent(agent, env, best_params, episodes=10000):
    """
    Train a PPO agent in the given environment using specified parameters.

    Args:
        agent: The PPO agent to be trained.
        env: The environment in which the agent will be trained.
        best_params (dict): A dictionary containing the best hyperparameters.
        episodes (int): The number of episodes to train the agent. Default is 10,000.

    Returns:
        list: A history of total rewards obtained in each episode.
    """
    reward_history = []  # To track total rewards per episode

    # Training loop over the specified number of episodes
    for episode in range(episodes):
        # Initialize buffers for experience storage
        states, actions, rewards, dones, old_probs = [], [], [], [], []
        state, _ = env.reset()  # Reset the environment to start the episode
        total_reward = 0  # Track total reward for this episode

        # Episode loop: collect data for 'buffer_size' steps or until the episode ends
        for t in range(best_params["buffer_size"]):
            # Convert the current state to a tensor for policy input
            state_tensor = torch.tensor(state, dtype=torch.float32)
            # Get action probabilities from the agent's policy
            action_probs = agent.policy(state_tensor).detach().numpy()
            # Sample an action based on the probabilities
            action = np.random.choice(env.action_space.n, p=action_probs)

            # Take the selected action in the environment
            new_state, reward, done, truncated, _ = env.step(action)

            # Store experience data in buffers
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done or truncated)
            old_probs.append(action_probs[action])

            # Update the state and accumulate rewards
            state = new_state
            total_reward += reward

            # Terminate the episode if done or truncated
            if done or truncated:
                break

        # Train the agent using the collected experience
        agent.train((states, actions, rewards, dones, old_probs))

        # Append total reward for the episode to the reward history
        reward_history.append(total_reward)

        # Print progress every 50 episodes
        if episode % 50 == 0:
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    return reward_history

def plot_rewards(reward_history):
    """
    Plot the total rewards per episode to visualize the agent's learning curve.

    Args:
        reward_history (list): A list of total rewards obtained in each episode during training.
    """
    # Create a plot for the learning curve
    plt.figure(figsize=(12, 6))
    plt.plot(reward_history, label="Total Reward")
    plt.xlabel("Episode")  # Label for the x-axis
    plt.ylabel("Total Reward")  # Label for the y-axis
    plt.title("Learning Curve (Custom PPO)")  # Title for the plot
    plt.grid()  # Add grid lines for better readability
    plt.legend()  # Add a legend to the plot
    plt.show()
