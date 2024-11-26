from stable_baselines3.common.callbacks import BaseCallback


class RewardLoggingCallback(BaseCallback):
    def __init__(self):
        super(RewardLoggingCallback, self).__init__()
        self.episode_rewards = []
        self.current_rewards = []

    def _on_step(self) -> bool:
        reward = self.locals["rewards"]
        done = self.locals["dones"]
        self.current_rewards.append(reward)

        if done:
            self.episode_rewards.append(sum(self.current_rewards))
            self.current_rewards = []
        return True
