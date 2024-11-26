import matplotlib.pyplot as plt

def evaluate_agent(env, agent):
    """
    Evaluate the trained agent and plot plasma concentration over time.

    Args:
        env: The wrapped environment for evaluation.
        agent: The trained PPO agent.

    Returns:
        state_trajectory: The trajectory of plasma concentration.
    """
    state_trajectory = []
    state, _ = env.reset()
    done = False

    while not done:
        state_trajectory.append(state[0])  # Record plasma concentration
        action, _ = agent.predict(state, deterministic=True)
        state, _, done, _, _ = env.step(action)

    # Plot Plasma Concentration Over Time
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(state_trajectory)), state_trajectory, label='Plasma Concentration', color='b')
    plt.axhline(y=10, color='g', linestyle='--', label='Therapeutic Lower Bound (10 mg/L)')
    plt.axhline(y=50, color='g', linestyle='--', label='Therapeutic Upper Bound (50 mg/L)')
    plt.axhline(y=100, color='r', linestyle='--', label='Toxic Threshold (>100 mg/L)')
    plt.xlabel('Time Step')
    plt.ylabel('Plasma Concentration (mg/L)')
    plt.title('Plasma Drug Concentration Over Time During Evaluation')
    plt.legend()
    plt.grid()
    plt.show()

    return state_trajectory
