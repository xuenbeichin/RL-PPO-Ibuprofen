import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt


def create_eval_env(video_folder, video_name_prefix):
    """
    Create an evaluation environment with video recording.

    This function initializes the CartPole-v1 environment and wraps it with
    a video recording wrapper that saves videos of all evaluation episodes.

    Args:
        video_folder (str): Folder where video recordings will be saved.
        video_name_prefix (str): Prefix for video file names.

    Returns:
        eval_env: The CartPole-v1 environment wrapped with a video recorder.
    """
    # Create the CartPole environment with RGB rendering for video recording
    eval_env = gym.make("CartPole-v1", render_mode="rgb_array")

    # Wrap the environment to record videos of all episodes
    eval_env = RecordVideo(
        eval_env,
        video_folder=video_folder,
        episode_trigger=lambda e: True,  # Record all episodes
        name_prefix=video_name_prefix
    )
    return eval_env


def evaluate_agent(final_model, eval_env, episodes=50):
    """
    Evaluate a trained PPO agent on the CartPole-v1 environment.

    This function runs the agent in the environment for a specified number of
    episodes and collects the state trajectory for analysis.

    Args:
        final_model: The trained PPO model to evaluate.
        eval_env: The evaluation environment.
        episodes (int): Number of episodes to run the evaluation. Default is 50.

    Returns:
        list: The state trajectory observed during the evaluation episodes.
    """
    state_trajectory = []  # To store the trajectory of states across episodes

    # Loop through the specified number of episodes
    for _ in range(episodes):
        state, _ = eval_env.reset()  # Reset the environment at the start of each episode
        done = False

        # Episode loop: interact with the environment until the episode ends
        while not done:
            state_trajectory.append(state)  # Record the current state
            action, _ = final_model.predict(state, deterministic=True)  # Get agent's action
            # Step the environment with the selected action
            state, reward, done, truncated, _ = eval_env.step(action)
            done = done or truncated  # Handle truncated episodes as done

    eval_env.close()  # Close the environment after evaluation
    return state_trajectory


def plot_pole_angles(state_trajectory):
    """
    Plot the pole angle over time using the state trajectory from evaluation.

    This function visualizes how the pole angle changes over time during the
    evaluation of the trained agent.

    Args:
        state_trajectory (list): A list of states observed during evaluation, where
                                 each state includes the pole angle as the third value.
    """
    # Extract pole angles (third element in each state)
    pole_angles = [s[2] for s in state_trajectory]

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(pole_angles)), pole_angles, label='Pole Angle', color='b')

    # Add a horizontal reference line for the vertical position
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, label="Vertical Position")

    plt.xlabel('Time Step')
    plt.ylabel('Pole Angle (radians)')
    plt.title('Pole Angle Over Time During Evaluation (SB3)')
    plt.legend()
    plt.grid()
    plt.show()
