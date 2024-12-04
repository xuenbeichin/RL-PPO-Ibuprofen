from stable_baselines3.common.callbacks import BaseCallback

class RewardLoggingCallback(BaseCallback):
    """
    Custom callback to log rewards during training using Stable-Baselines3.

    This callback accumulates rewards for each episode and logs them when
    the episode ends. It can be used to track the agent's performance over time.
    """

    def __init__(self):
        """
        Initialize the RewardLoggingCallback.
        """
        super(RewardLoggingCallback, self).__init__()
        self.episode_rewards = []  # List to store the total rewards per episode
        self.current_episode_reward = 0  # Running total of rewards for the current episode

    def _on_step(self) -> bool:
        """
        Method called at each environment step.

        Tracks rewards and resets the reward counter when an episode ends.

        Returns:
            bool: Always returns True to indicate the training process should continue.
        """
        # Access the reward for the current step
        reward = self.locals["rewards"][0]
        # Check if the current step ends the episode
        done = self.locals["dones"][0]
        # Accumulate the reward for the current episode
        self.current_episode_reward += reward

        # If the episode is done, log the total reward and reset the counter
        if done:
            self.episode_rewards.append(self.current_episode_reward)  # Log the reward
            self.current_episode_reward = 0  # Reset for the next episode

        # Return True to continue training
        return True
