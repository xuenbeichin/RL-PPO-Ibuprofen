
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DrugDeliveryEnv(gym.Env):
    def __init__(self):
        super(DrugDeliveryEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)  # Drug dosage
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)  # Drug concentration

        self.target_concentration = 0.7  # Target drug concentration
        self.state = np.array([0.0])  # Initial concentration
        self.time_step = 0
        self.max_time_steps = 100

    def reset(self, seed=None, options=None):
        # Seed the environment for reproducibility
        if seed is not None:
            np.random.seed(seed)

        self.state = np.array([0.0])
        self.time_step = 0
        return self.state, {}

    def step(self, action):
        # Simulate drug absorption dynamics
        action = np.clip(action, 0.0, 1.0)
        current_concentration = self.state[0]
        absorption = action[0] * 0.5
        decay = current_concentration * 0.1
        new_concentration = current_concentration + absorption - decay

        self.state = np.array([np.clip(new_concentration, 0.0, 1.0)])
        reward = -abs(self.state[0] - self.target_concentration)
        self.time_step += 1

        done = self.time_step >= self.max_time_steps
        return self.state, reward, done, {}

    def render(self, mode="human"):
        print(f"Time Step: {self.time_step}, Concentration: {self.state[0]:.3f}")
