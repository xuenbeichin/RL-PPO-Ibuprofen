import matplotlib.pyplot as plt


def evaluate_agent(env, model):
    """
    Evaluate a trained agent in the given environment and record the plasma concentration over time.

    Args:
        env: The environment in which the agent will be evaluated.
        model: The trained RL model to evaluate.

    Returns:
        list: A trajectory of plasma concentrations observed during the evaluation episode.
    """
    state_trajectory = []  # List to store the plasma concentration trajectory
    state = env.reset()  # Reset the environment to the initial state
    done = False  # Flag to track the end of the episode

    # Run the evaluation loop
    while not done:
        state_trajectory.append(state[0])  # Record plasma concentration (assumes first state value)
        action, _ = model.predict(state, deterministic=True)  # Predict action from the model
        state, reward, done, info = env.step(action)  # Take action in the environment

    return state_trajectory  # Return the recorded state trajectory


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
