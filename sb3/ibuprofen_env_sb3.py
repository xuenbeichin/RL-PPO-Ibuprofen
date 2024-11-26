import gymnasium as gym
import numpy as np

class IbuprofenEnv(gym.Env):
    def __init__(self):
        super(IbuprofenEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)

        # Pharmacokinetics parameters
        self.therapeutic_range = (10, 50)  # Therapeutic range for the drug (mg/L)
        self.half_life = 2.0  # Plasma half-life in hours
        self.clearance_rate = 0.693 / self.half_life  # First-order decay constant
        self.time_step_hours = 6  # Each time step represents 6 hours

        self.max_steps = 24  # 24 time steps = 6-hour intervals over 6 days
        self.current_step = 0
        self.plasma_concentration = 0.0
        self.np_random = None  # RNG for seeding

    def reset(self, seed=None, options=None):
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.current_step = 0
        self.plasma_concentration = 0.0
        return np.array([self.plasma_concentration], dtype=np.float32), {}

    def step(self, action):
        dose_mg = action * 200  # Map action index to dose in mg
        absorbed = dose_mg / 10  # Simplified absorption model
        self.plasma_concentration += absorbed
        self.plasma_concentration *= np.exp(-self.clearance_rate * self.time_step_hours)

        # Calculate the reward
        if self.therapeutic_range[0] <= self.plasma_concentration <= self.therapeutic_range[1]:
            reward = 10
        elif self.plasma_concentration > 100:
            reward = -20
        elif self.plasma_concentration < self.therapeutic_range[0]:
            reward = -5
        else:
            reward = -10

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        info = {}

        return (
            np.array([self.plasma_concentration], dtype=np.float32),
            reward,
            terminated,
            truncated,
            info,
        )
