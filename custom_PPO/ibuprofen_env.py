# Define the Ibuprofen Environment
import gymnasium as gym
import numpy as np


class IbuprofenEnv(gym.Env):
    """
    Custom OpenAI Gym environment to simulate ibuprofen plasma concentration.

    The environment models the effect of ibuprofen dosing over time, considering
    pharmacokinetics such as absorption and clearance. The agent's goal is to
    keep the plasma concentration within the therapeutic range.

    Attributes:
        action_space (gym.spaces.Discrete): Action space with 5 discrete actions.
            Actions correspond to doses: 0 (no dose), 1 (200 mg), ..., 4 (800 mg).
        observation_space (gym.spaces.Box): State space representing plasma concentration.
        therapeutic_range (tuple): The target plasma concentration range (10-50 mg/L).
        half_life (float): Plasma half-life of ibuprofen in hours.
        clearance_rate (float): Clearance rate based on the half-life.
        time_step_hours (int): Time represented by one simulation step in hours.
        max_steps (int): Maximum number of steps per episode.
        current_step (int): Counter for the current step in the episode.
        plasma_concentration (float): The current plasma concentration (mg/L).
    """

    def __init__(self):
        """
        Initializes the Ibuprofen environment.
        """
        super(IbuprofenEnv, self).__init__()

        # Define the action space: 5 discrete actions (0 mg to 800 mg).
        self.action_space = gym.spaces.Discrete(5)

        # Define the observation space: Plasma concentration between 0 and 100 mg/L.
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)

        # Pharmacokinetics parameters
        self.therapeutic_range = (10, 50)  # Target therapeutic range in mg/L.
        self.half_life = 2.0  # Plasma half-life in hours.
        self.clearance_rate = 0.693 / self.half_life  # First-order decay constant.
        self.time_step_hours = 6  # Each simulation step represents 6 hours.

        # Simulation settings
        self.max_steps = 24  # Maximum number of steps (e.g., 6-hour intervals over 6 days).
        self.current_step = 0  # Current step in the episode.
        self.plasma_concentration = 0.0  # Initial plasma concentration.

    def reset(self):
        """
        Resets the environment to its initial state.

        Returns:
            np.ndarray: The initial plasma concentration as a single-element array.
        """
        self.current_step = 0
        self.plasma_concentration = 0.0  # Reset plasma concentration to zero.
        return np.array([self.plasma_concentration], dtype=np.float32)

    def step(self, action):
        """
        Simulates one step in the environment based on the chosen action.

        Args:
            action (int): The chosen dose, represented as an action index (0-4).

        Returns:
            tuple:
                - np.ndarray: The updated plasma concentration as a single-element array.
                - float: The reward for the current step based on therapeutic range.
                - bool: Whether the episode has ended.
                - dict: Additional information (empty in this case).
        """
        # Convert action to dose (0-800 mg based on action index).
        dose_mg = action * 200

        # Simulate absorption (10% bioavailability).
        absorbed = dose_mg / 10
        self.plasma_concentration += absorbed

        # Simulate clearance using exponential decay.
        self.plasma_concentration *= np.exp(-self.clearance_rate * self.time_step_hours)

        # Determine the reward based on the current plasma concentration.
        if self.therapeutic_range[0] <= self.plasma_concentration <= self.therapeutic_range[1]:
            reward = 10  # High reward for staying in the therapeutic range.
        elif self.plasma_concentration > 100:
            reward = -20  # Penalty for exceeding toxic levels (>100 mg/L).
        elif self.plasma_concentration < self.therapeutic_range[0]:
            reward = -5  # Penalty for being subtherapeutic (<10 mg/L).
        else:
            reward = -10  # Penalty for being above therapeutic but below toxic.

        # Increment the step counter and check if the episode is done.
        self.current_step += 1
        done = self.current_step >= self.max_steps  # Episode ends after max steps.

        # Return the updated state, reward, and done flag.
        return np.array([self.plasma_concentration], dtype=np.float32), reward, done, {}
