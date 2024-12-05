import gymnasium as gym
import numpy as np
import optuna
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
from tensorflow.python.summary.writer import writer


# Environment Definition
class IbuprofenEnv(gym.Env):
    def __init__(self, normalize=False):
        super(IbuprofenEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(5)  # Actions: discrete doses (0-4 units, each unit = 200 mg)
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)
        self.therapeutic_range = (10, 50)  # Target therapeutic range for the drug
        self.half_life = 2.0  # Half-life of the drug in hours
        self.clearance_rate = 0.693 / self.half_life
        self.time_step_hours = 1
        self.bioavailability = 0.9
        self.volume_of_distribution = 0.15  # L/kg
        self.max_steps = 24
        self.current_step = 0
        self.plasma_concentration = 0.0
        self.normalize = normalize
        self.state_buffer = []

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self.current_step = 0
        self.plasma_concentration = 0.0
        self.state_buffer = []
        state = np.array([self.plasma_concentration], dtype=np.float32)
        return self._normalize(state), {}

    def step(self, action):
        dose_mg = action * 200
        absorbed_mg = dose_mg * self.bioavailability
        absorbed_concentration = absorbed_mg / (self.volume_of_distribution * 70)
        self.plasma_concentration += absorbed_concentration
        self.plasma_concentration *= np.exp(-self.clearance_rate * self.time_step_hours)

        state = np.array([self.plasma_concentration], dtype=np.float32)
        normalized_state = self._normalize(state)
        self.state_buffer.append(self.plasma_concentration)

        reward = 0
        if self.therapeutic_range[0] <= self.plasma_concentration <= self.therapeutic_range[1]:
            reward = 10
        elif self.plasma_concentration < self.therapeutic_range[0]:
            reward = -5 - (self.therapeutic_range[0] - self.plasma_concentration) * 0.5
        elif self.plasma_concentration > self.therapeutic_range[1]:
            reward = -5 - (self.plasma_concentration - self.therapeutic_range[1]) * 0.5
        if self.plasma_concentration > 100:
            reward -= 15

        self.current_step += 1
        done = self.current_step >= self.max_steps
        truncated = False
        return normalized_state, reward, done, truncated, {}

    def _normalize(self, state):
        if self.normalize and len(self.state_buffer) > 1:
            mean = np.mean(self.state_buffer)
            std = np.std(self.state_buffer) + 1e-8
            return (state - mean) / std
        return state

# Custom Callbacks
class RewardLoggingCallback(BaseCallback):
    def __init__(self):
        super(RewardLoggingCallback, self).__init__()
        self.episode_rewards = []
        self.current_episode_reward = 0

    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_episode_reward)
            if self.logger:
                self.logger.record("reward/episode_reward", self.current_episode_reward)
            self.current_episode_reward = 0
        return True

class TensorBoardMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorBoardMetricsCallback, self).__init__(verbose)
        self.tb_writer = None

    def _on_training_start(self):
        if self.logger:
            self.tb_writer = self.logger.output_formats[0].writer

    def _on_step(self) -> bool:
        metrics = self.model.logger.name_to_value
        if self.tb_writer:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(f"Metrics/{key}", value, self.num_timesteps)
        return True

# Hyperparameter Optimization with Optuna
def optimize_ppo(trial):
    env = DummyVecEnv([lambda: IbuprofenEnv(normalize=True)])
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.99)
    n_epochs = trial.suggest_int("n_epochs", 3, 10)
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 32, 512, step=32)
    n_steps = trial.suggest_int("n_steps", 64, 2048, step=64)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr,
        gamma=gamma,
        n_epochs=n_epochs,
        ent_coef=ent_coef,
        batch_size=batch_size,
        n_steps=n_steps,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        verbose=0,
    )
    model.learn(total_timesteps=100000)

    rewards = []
    for _ in range(10):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)

    return np.mean(rewards)

# Optimize Hyperparameters
study = optuna.create_study(direction="maximize")
study.optimize(optimize_ppo, n_trials=100)
best_params = study.best_params
print("Best Parameters:", best_params)

# Configure TensorBoard Logger
log_dir = "./tensorboard_logs10000/"
os.makedirs(log_dir, exist_ok=True)
custom_logger = configure(log_dir, ["tensorboard"])

# Train Final Model
env = DummyVecEnv([lambda: IbuprofenEnv(normalize=True)])
final_model = PPO(
    "MlpPolicy",
    env,
    learning_rate=best_params["learning_rate"],
    gamma=best_params["gamma"],
    n_epochs=best_params["n_epochs"],
    ent_coef=best_params["ent_coef"],
    batch_size=best_params["batch_size"],
    n_steps=best_params["n_steps"],
    gae_lambda=best_params["gae_lambda"],
    clip_range=best_params["clip_range"],
    verbose=1,
    tensorboard_log=log_dir,
)

callback = CallbackList([TensorBoardMetricsCallback(), RewardLoggingCallback()])
final_model.set_logger(custom_logger)
final_model.learn(total_timesteps=100000, callback=callback)

# Plot Learning Curve
reward_callback = RewardLoggingCallback()
final_model.learn(total_timesteps=100000, callback=reward_callback)


plt.figure(figsize=(12, 6))
plt.plot(reward_callback.episode_rewards, label="Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Learning Curve")
plt.grid()
plt.legend()
plt.show()



# Evaluation Loop
evaluation_episodes = 100000  # Number of episodes for evaluation

evaluation_rewards = []
plasma_concentration_trajectories = []

# Access the underlying environment from DummyVecEnv
env = env.envs[0]  # envs[0] gives access to the unwrapped IbuprofenEnv

for episode in range(evaluation_episodes):
    state, _ = env.reset()

    total_reward = 0
    plasma_concentration_history = [state[0]]  # Track plasma concentration

    for _ in range(env.max_steps):  # Use max_steps from the underlying environment
        # Use the SB3 predict method for actions
        action, _ = final_model.predict(state, deterministic=True)

        # Take the chosen action in the environment
        new_state, reward, done, truncated, _ = env.step(action)
        plasma_concentration_history.append(new_state[0])

        state = new_state
        total_reward += reward

        if done or truncated:
            break

    evaluation_rewards.append(total_reward)
    plasma_concentration_trajectories.append(plasma_concentration_history)

# Plot plasma concentration from the last evaluation episode
plt.figure(figsize=(12, 6))
plt.plot(plasma_concentration_trajectories[-1], label="Plasma Concentration")
plt.axhline(y=env.therapeutic_range[0], color="g", linestyle="--", label="Lower Therapeutic Range")
plt.axhline(y=env.therapeutic_range[1], color="g", linestyle="--", label="Upper Therapeutic Range")
plt.axhline(y=100, color="r", linestyle="--", label="Toxic Level")
plt.xlabel("Time (hours)")
plt.ylabel("Plasma Concentration (mg/L)")
plt.title("Plasma Concentration Over Time (SB3)")
plt.legend()
plt.grid()
plt.show()
