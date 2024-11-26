import numpy as np
import torch
import matplotlib.pyplot as plt
from PPO.PPO_agent import PPOAgent
from run import IbuprofenEnv


def train_agent(agent, env, episodes=1000):
    """
    Train the PPO agent using the provided environment for a specified number of episodes.

    Args:
        agent: The PPO agent to train.
        env: The environment to interact with.
        episodes (int): The number of episodes to train the agent.

    Returns:
        list: History of rewards during training.
    """
    reward_history = []

    for episode in range(episodes):
        state = env.reset()  # Reset environment for each episode
        states, actions, rewards, dones, old_probs = [], [], [], [], []  # Initialize buffers
        total_reward = 0  # Track total reward for the episode

        while True:
            # Select action based on policy
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_probs = agent.policy(state_tensor).detach().numpy()
            action = np.random.choice(env.action_space.n, p=action_probs)

            # Take action in the environment
            new_state, reward, done, _ = env.step(action)

            # Store transition data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            old_probs.append(action_probs[action])

            state = new_state  # Update state
            total_reward += reward
            if done:  # End episode if done
                break

        # Train the agent using the collected data
        agent.train(states, actions, rewards, dones, old_probs)
        reward_history.append(total_reward)  # Track episode rewards
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    return reward_history

def plot_training(reward_history):
    plt.figure(figsize=(12, 6))
    plt.plot(reward_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('PPO Training Performance for Ibuprofen Delivery')
    plt.grid()
    plt.show()



if __name__ == "__main__":

    # Initialize environment and agent
    env = IbuprofenEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)

    # Train the agent
    reward_history = train_agent(agent, env, episodes=1000)

    plot_training(reward_history)