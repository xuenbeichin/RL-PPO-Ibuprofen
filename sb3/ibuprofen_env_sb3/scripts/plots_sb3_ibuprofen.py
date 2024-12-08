from matplotlib import pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv

from environments.ibuprofen_env import IbuprofenEnv


def plot_reward_history(reward_history, title="Learning Curve (SB3)"):
    """
    Plots the reward history to visualize progress over episodes.

    Args:
        reward_history (list): A list of total rewards for each episode.
        title (str, optional): Title of the plot. Defaults to "Learning Curve (SB3)".
    """
    plt.figure(figsize=(12, 6))
    plt.plot(reward_history, label="Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_last_episode_concentration(env, model, therapeutic_range=(10, 50), toxic_threshold=100):
    """
    Plots the plasma concentration over time for the last episode.

    Args:
        env (DummyVecEnv): The environment used for evaluation.
        model (PPO): The trained PPO model.
        therapeutic_range (tuple): The therapeutic range (min, max) for the drug concentration.
        toxic_threshold (float): The toxic threshold for the drug concentration.
    """
    obs = env.reset()  # Reset the environment
    plasma_concentrations = []  # List to store plasma concentrations over time
    done = False

    while not done:
        # Predict action using the trained model
        action, _ = model.predict(obs, deterministic=True)
        # Take a step in the environment
        obs, _, done, info = env.step(action)
        # Collect the current plasma concentration
        plasma_concentrations.append(env.envs[0].plasma_concentration)

    # Plot the plasma concentration
    plt.figure(figsize=(12, 6))
    plt.plot(plasma_concentrations, label="Plasma Concentration")
    plt.axhline(y=therapeutic_range[0], color='green', linestyle='--', label="Therapeutic Min")
    plt.axhline(y=therapeutic_range[1], color='green', linestyle='--', label="Therapeutic Max")
    plt.axhline(y=toxic_threshold, color='red', linestyle='--', label="Toxic Threshold")
    plt.xlabel("Time Step")
    plt.ylabel("Plasma Concentration")
    plt.title("Plasma Concentration Over Time (Last Episode)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Plot the reward history
    plot_reward_history(reward_history)

    # Evaluate and plot the plasma concentration for the last episode
    env = DummyVecEnv([lambda: IbuprofenEnv(normalize=False)])  # Use the original, non-normalized environment
    plot_last_episode_concentration(env, final_model)
