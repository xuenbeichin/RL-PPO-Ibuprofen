# Plot training performance
from matplotlib import pyplot as plt
import numpy as np
import torch
from PPO.PPO_agent import PPOAgent
from run import IbuprofenEnv

def evaluate_agent(agent, env, episodes=1000):
    """
    Evaluate the trained PPO agent by running it in the environment over multiple episodes and visualizing its performance.

    Args:
        agent: The PPO agent to evaluate.
        env: The environment to interact with.
        episodes (int): The number of episodes to run for evaluation.
    """
    avg_trajectory = []  # To store average plasma concentration over multiple episodes
    for episode in range(episodes):
        state_trajectory = []  # Store plasma concentration for each episode
        state = env.reset()
        done = False

        while not done:
            state_trajectory.append(state[0])  # Record plasma concentration for each step
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action = agent.policy(state_tensor).argmax().item()  # Greedy action selection
            state, reward, done, _ = env.step(action)

        avg_trajectory.append(state_trajectory)  # Add the trajectory for this episode

    # Calculate average plasma concentration over episodes
    avg_plasma_concentration = [np.mean([trajectory[t] for trajectory in avg_trajectory]) for t in range(len(avg_trajectory[0]))]

    # Plot plasma concentration over time (average across episodes)
    plt.figure(figsize=(12, 6))
    plt.plot(avg_plasma_concentration, label='Average Plasma Concentration', color='b')
    plt.axhline(y=10, color='g', linestyle='--', label='Therapeutic Lower Bound (10 mg/L)')
    plt.axhline(y=50, color='g', linestyle='--', label='Therapeutic Upper Bound (50 mg/L)')
    plt.axhline(y=100, color='r', linestyle='--', label='Toxic Threshold (>100 mg/L)')
    plt.xlabel('Time Step')
    plt.ylabel('Plasma Concentration (mg/L)')
    plt.title(f'Plasma Drug Concentration Over Time (Average of {episodes} Episodes)')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":

    # Initialize environment and agent
    env = IbuprofenEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)

    evaluate_agent(agent, env, episodes=10)