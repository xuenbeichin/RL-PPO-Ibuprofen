import torch
import matplotlib.pyplot as plt
from ibuprofen_env import IbuprofenEnv
from ppo_agent import PPOAgent
from hyperparameter_tuning import run_hyperparameter_tuning

def evaluate_agent(agent, env):
    """
    Evaluates the trained agent in the Ibuprofen environment.

    The agent interacts with the environment, and the trajectory of plasma
    concentrations is recorded for visualization.

    Args:
        agent (PPOAgent): The trained PPO agent.
        env (gym.Env): The Ibuprofen environment.

    Returns:
        list: Plasma concentration trajectory during the evaluation episode.
    """
    # Reset the environment to its initial state
    state = env.reset()
    state_trajectory = []  # To record plasma concentration at each step
    done = False

    while not done:
        state_trajectory.append(state[0])  # Record the current plasma concentration
        state_tensor = torch.tensor(state, dtype=torch.float32)  # Convert state to a tensor
        action = agent.policy(state_tensor).argmax().item()  # Get the action with the highest probability
        state, reward, done, _ = env.step(action)  # Take the action in the environment

    return state_trajectory

def plot_evaluation(state_trajectory):
    """
    Plots the plasma concentration trajectory over time.

    The plot includes the therapeutic range and toxic threshold for visual reference.

    Args:
        state_trajectory (list): Plasma concentrations at each time step.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(state_trajectory)), state_trajectory, label='Plasma Concentration', color='b')
    plt.axhline(y=10, color='g', linestyle='--', label='Therapeutic Lower Bound (10 mg/L)')
    plt.axhline(y=50, color='g', linestyle='--', label='Therapeutic Upper Bound (50 mg/L)')
    plt.axhline(y=100, color='r', linestyle='--', label='Toxic Threshold (>100 mg/L)')
    plt.xlabel('Time Step')
    plt.ylabel('Plasma Concentration (mg/L)')
    plt.title('Plasma Drug Concentration Over Time')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":

    # Step 1: Perform hyperparameter tuning to find the best parameters
    best_params = run_hyperparameter_tuning(n_trials=20)

    # Step 2: Initialize the environment and the PPO agent
    env = IbuprofenEnv()
    state_dim = env.observation_space.shape[0]  # Dimension of the state space
    action_dim = env.action_space.n  # Number of possible actions

    # Create a PPO agent with the optimized hyperparameters
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, **best_params)

    # Step 3: Evaluate the trained agent
    trajectory = evaluate_agent(agent, env)

    # Step 4: Plot the evaluation results
    plot_evaluation(trajectory)
