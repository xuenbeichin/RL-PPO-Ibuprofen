import numpy as np
import gym

class IbuprofenEnv(gym.Env):
    """
    Custom environment for simulating the pharmacokinetics of ibuprofen in the human body.

    This environment models the plasma concentration of ibuprofen over time, considering the drug's absorption
    and clearance from the body. The agent can take actions to administer doses of ibuprofen and the goal is to
    maintain the plasma concentration within a therapeutic range while avoiding toxicity.

    Attributes:
        action_space (gym.spaces.Discrete): Discrete action space where:
            0 -> No dose, 1 -> 200 mg, 2 -> 400 mg, 3 -> 600 mg, 4 -> 800 mg.
        observation_space (gym.spaces.Box): Continuous state space representing plasma concentration (mg/L).
        therapeutic_range (tuple): The therapeutic plasma concentration range (10-50 mg/L).
        half_life (float): The half-life of ibuprofen in hours (2.0 hours).
        clearance_rate (float): The first-order decay rate constant calculated from the half-life.
        time_step_hours (int): Time step duration in hours (6 hours).
        max_steps (int): The maximum number of time steps per episode (24 steps, i.e., 6 days).
        current_step (int): The current time step in the simulation.
        plasma_concentration (float): The current plasma concentration of ibuprofen (mg/L).
    """

    def __init__(self):
        """
        Initialize the Ibuprofen environment.

        Sets up the action space, observation space, pharmacokinetics parameters, and simulation settings.
        """
        super(IbuprofenEnv, self).__init__()
        # Action space: 0 (No dose), 1 (200 mg), ..., 4 (800 mg)
        self.action_space = gym.spaces.Discrete(5)
        # State space: plasma concentration (mg/L)
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)

        # Pharmacokinetics parameters
        self.therapeutic_range = (10, 50)  # Therapeutic range for the drug (mg/L)
        self.half_life = 2.0  # Plasma half-life in hours
        self.clearance_rate = 0.693 / self.half_life  # First-order decay constant
        self.time_step_hours = 6  # Each time step represents 6 hours

        # Simulation settings
        self.max_steps = 24  # 24 time steps = 6-hour intervals over 6 days
        self.current_step = 0
        self.plasma_concentration = 0.0  # Initial plasma concentration

    def reset(self):
        """
        Reset the environment to its initial state.

        Resets the plasma concentration and simulation step count to their initial values.
        Returns the initial state (plasma concentration) as a numpy array.

        Returns:
            np.ndarray: Initial plasma concentration (0.0 mg/L).
        """
        self.current_step = 0
        self.plasma_concentration = 0.0
        return np.array([self.plasma_concentration], dtype=np.float32)

    def step(self, action):
        """
        Take an action (dose) and return the new state, reward, done flag, and info dictionary.

        The action corresponds to a dose of ibuprofen (0-4), which is applied to the current plasma concentration.
        The reward is based on how well the plasma concentration matches the therapeutic range.

        Args:
            action (int): The action taken by the agent (0-4 corresponding to doses).

        Returns:
            tuple:
                - np.ndarray: The new plasma concentration (state).
                - float: The reward based on the plasma concentration.
                - bool: Whether the episode is done (reached max steps).
                - dict: An empty dictionary for additional information.
        """
        # Determine dose based on action
        dose_mg = action * 200  # Action index maps to dose in mg (0 -> 0 mg, 1 -> 200 mg, ..., 4 -> 800 mg)
        absorbed = dose_mg / 10  # Assume 10% bioavailability (simplified absorption model)
        self.plasma_concentration += absorbed

        # Clearance via exponential decay (pharmacokinetics)
        self.plasma_concentration *= np.exp(-self.clearance_rate * self.time_step_hours)

        # Calculate reward based on plasma concentration
        if self.therapeutic_range[0] <= self.plasma_concentration <= self.therapeutic_range[1]:
            reward = 10  # Reward for being within the therapeutic range
        elif self.plasma_concentration > 100:  # Toxic concentration
            reward = -20  # Heavy penalty for toxicity
        elif self.plasma_concentration < self.therapeutic_range[0]:
            reward = -5  # Penalty for being subtherapeutic
        else:
            reward = -10  # Penalty for exceeding therapeutic but below toxic levels

        # Update simulation step
        self.current_step += 1
        done = self.current_step >= self.max_steps  # Episode ends after max steps

        return np.array([self.plasma_concentration], dtype=np.float32), reward, done, {}
