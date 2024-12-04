import torch
import numpy as np
from matplotlib import pyplot as plt


import torch
import numpy as np

def evaluate_agent(agent, env):
    """
    Evaluate a trained agent in the given environment and track the plasma concentration.

    This function interacts with the environment using the agent's policy to determine actions
    and records the plasma concentration at each time step.

    Args:
        agent: The trained agent to be evaluated. The agent should have a `policy` attribute
               that provides action probabilities given a state.
        env: The environment in which the agent will be evaluated. The environment must have
             `reset()` and `step()` methods, and a `max_steps` attribute.

    Returns:
        list: A history of plasma concentrations recorded during the evaluation.
    """
    # Reset the environment and get the initial state
    state, _ = env.reset()  # Reset the environment and get the initial state
    plasma_concentration_history = [state[0]]  # Initialize the history with the initial plasma concentration

    # Run the evaluation loop for a maximum number of steps
    for t in range(env.max_steps):  # Use the maximum steps defined by the environment
        # Convert the state into a tensor for the agent's policy network
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension (1, state_dim)

        # Predict action probabilities using the agent's policy
        action_probs = agent.policy(state_tensor).detach().numpy().flatten()
        # Select the action with the highest probability
        action = np.argmax(action_probs)

        # Step the environment with the selected action
        new_state, _, done, _, _ = env.step(action)

        # Record the plasma concentration from the new state
        plasma_concentration_history.append(new_state[0])
        state = new_state  # Update the state for the next step

        # End the evaluation if the environment signals `done`
        if done:
            break

    return plasma_concentration_history



def plot_plasma_concentration(state_trajectory):
    """
    Plot the plasma concentration over time with therapeutic and toxic thresholds.

    Args:
        state_trajectory (list): The trajectory of plasma concentrations recorded during evaluation.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(
        range(len(state_trajectory)),
        state_trajectory,
        label="Plasma Concentration",
        color="b"
    )

    # Add therapeutic and toxic thresholds
    plt.axhline(y=10, color="g", linestyle="--", label="Therapeutic Lower Bound (10 mg/L)")
    plt.axhline(y=50, color="g", linestyle="--", label="Therapeutic Upper Bound (50 mg/L)")
    plt.axhline(y=100, color="r", linestyle="--", label="Toxic Threshold (>100 mg/L)")

    plt.xlabel("Time Step")
    plt.ylabel("Plasma Concentration (mg/L)")
    plt.title("Plasma Drug Concentration Over Time (SB3)")
    plt.legend()
    plt.grid()
    plt.show()