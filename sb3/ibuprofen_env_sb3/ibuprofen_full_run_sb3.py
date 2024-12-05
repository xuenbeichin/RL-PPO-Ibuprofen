import gymnasium as gym
import numpy as np
import optuna
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt


class IbuprofenEnv(gym.Env):
    def __init__(self, normalize=False):
        super(IbuprofenEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)
        self.therapeutic_range = (10, 50)
        self.half_life = 2.0
        self.clearance_rate = 0.693 / self.half_life
        self.time_step_hours = 1
        self.bioavailability = 0.9
        self.volume_of_distribution = 0.15
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

        if self.therapeutic_range[0] <= self.plasma_concentration <= self.therapeutic_range[1]:
            reward = 10
        else:
            if self.plasma_concentration < self.therapeutic_range[0]:
                reward = -5 - (self.therapeutic_range[0] - self.plasma_concentration) * 0.5
            elif self.plasma_concentration > self.therapeutic_range[1]:
                reward = -5 - (self.plasma_concentration - self.therapeutic_range[1]) * 0.5

        if self.plasma_concentration > 100:
            reward -= 15

        self.current_step += 1
        done = self.current_step >= self.max_steps
        truncated = False
        info = {}

        return normalized_state, reward, done, truncated, info

    def _normalize(self, state):
        if self.normalize and len(self.state_buffer) > 1:
            mean = np.mean(self.state_buffer)
            std = np.std(self.state_buffer) + 1e-8
            return (state - mean) / std
        return state


class RewardLoggingCallback(BaseCallback):
    """
    Custom callback for logging rewards and additional PPO-specific metrics to TensorBoard.
    """
    def __init__(self):
        super(RewardLoggingCallback, self).__init__()
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Accumulate rewards for the current episode
        self.current_episode_reward += self.locals["rewards"][0]

        # Log rewards when the episode ends
        if self.locals["dones"][0]:
            self.logger.record("episode/reward", self.current_episode_reward)
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_count += 1
            self.current_episode_reward = 0  # Reset for the next episode

        return True

    def _on_rollout_end(self) -> None:
        """
        Log additional PPO-specific metrics at the end of a rollout.
        These metrics are computed during the update step in PPO.
        """
        try:
            metrics = {
                "train/approxkl": self.locals.get("approx_kl", None),
                "train/clipfrac": self.locals.get("clip_fraction", None),
                "train/entropy_loss": self.locals.get("entropy_loss", None),
                "train/explained_variance": self.locals.get("explained_variance", None),
                "train/fps": self.locals.get("fps", None),
                "train/policy_loss": self.locals.get("policy_loss", None),
                "train/value_loss": self.locals.get("value_loss", None),
                "train/loss": self.locals.get("loss", None),  # Total loss
            }
            # Record metrics in TensorBoard
            for key, value in metrics.items():
                if value is not None:
                    self.logger.record(key, value)
        except KeyError:
            # Skip if the metrics are not available
            pass

class TensorBoardCallback(BaseCallback):
    def __init__(self, writer):
        super(TensorBoardCallback, self).__init__()
        self.writer = writer
        self.episode_rewards = []
        self.current_episode_reward = 0

    def _on_step(self) -> bool:
        # Track rewards for the current episode
        self.current_episode_reward += self.locals["rewards"][0]

        # If the episode ends, log the reward
        if self.locals["dones"][0]:
            episode = len(self.episode_rewards)
            self.episode_rewards.append(self.current_episode_reward)
            self.writer.add_scalar("Reward/Episode", self.current_episode_reward, episode)
            self.current_episode_reward = 0

        # Log metrics during training
        self.writer.add_scalar("Metrics/approxkl", self.locals.get("approx_kl", 0), self.num_timesteps)
        self.writer.add_scalar("Metrics/clipfrac", self.locals.get("clipfrac", 0), self.num_timesteps)
        self.writer.add_scalar("Metrics/explained_variance", self.locals.get("explained_variance", 0), self.num_timesteps)
        self.writer.add_scalar("Metrics/policy_entropy", self.locals.get("entropy", 0), self.num_timesteps)
        self.writer.add_scalar("Metrics/policy_loss", self.locals.get("pg_loss", 0), self.num_timesteps)
        self.writer.add_scalar("Metrics/value_loss", self.locals.get("value_loss", 0), self.num_timesteps)

        # Log any additional metrics
        return True

    def _on_training_end(self):
        self.writer.close()


def optimize_ppo(trial):
    """
    Optimize PPO hyperparameters using Optuna.
    """
    # Environment
    env = DummyVecEnv([lambda: IbuprofenEnv(normalize=True)])

    # Hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.99)
    n_epochs = trial.suggest_int("n_epochs", 3, 10)
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 32, 512, step=32)
    n_steps = trial.suggest_int("n_steps", 64, 2048, step=64)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)

    # Model
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

    # Train the model for a sufficient number of timesteps
    model.learn(total_timesteps=10000)  # Increased training timesteps

    # Evaluation
    rewards = []
    for _ in range(10):  #
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)

    # Return the mean reward over evaluation episodes
    return np.mean(rewards)


# Run Optuna optimization
study = optuna.create_study(direction="maximize")
study.optimize(optimize_ppo, n_trials=100)

best_params = study.best_params
print("Best Parameters:", best_params)

# TensorBoard logging directory
log_dir = "./tensorboard/ppo_ibuprofen"
new_logger = configure(log_dir, ["tensorboard"])

# Final model training
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
)
final_model.set_logger(new_logger)

# Train with custom callback
callback = RewardLoggingCallback()
final_model.learn(total_timesteps=10000, callback=callback)

# Plot episode rewards
plt.figure(figsize=(12, 6))
plt.plot(callback.episode_rewards, label="Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Learning Curve (SB3)")
plt.legend()
plt.grid()
plt.show()



# Evaluation Loop
evaluation_episodes = 100  # Number of episodes for evaluation

evaluation_rewards = []
plasma_concentration_trajectories = []

# Access the underlying environment from DummyVecEnv
underlying_env = env.envs[0]  # envs[0] gives access to the unwrapped IbuprofenEnv

for episode in range(evaluation_episodes):
    state, _ = underlying_env.reset()

    total_reward = 0
    plasma_concentration_history = [state[0]]  # Track plasma concentration

    for _ in range(underlying_env.max_steps):  # Use max_steps from the underlying environment
        # Use the SB3 predict method for actions
        action, _ = final_model.predict(state, deterministic=True)

        # Take the chosen action in the environment
        new_state, reward, done, truncated, _ = underlying_env.step(action)
        plasma_concentration_history.append(new_state[0])

        state = new_state
        total_reward += reward

        if done or truncated:
            break

    evaluation_rewards.append(total_reward)
    plasma_concentration_trajectories.append(plasma_concentration_history)

# Access the underlying environment from DummyVecEnv
underlying_env = env.envs[0]

# Plot plasma concentration from the last evaluation episode
plt.figure(figsize=(12, 6))
plt.plot(plasma_concentration_trajectories[-1], label="Plasma Concentration")
plt.axhline(y=underlying_env.therapeutic_range[0], color="g", linestyle="--", label="Lower Therapeutic Range")
plt.axhline(y=underlying_env.therapeutic_range[1], color="g", linestyle="--", label="Upper Therapeutic Range")
plt.axhline(y=100, color="r", linestyle="--", label="Toxic Level")
plt.xlabel("Time (hours)")
plt.ylabel("Plasma Concentration (mg/L)")
plt.title("Plasma Concentration Over Time (SB3)")
plt.legend()
plt.grid(True)
plt.show()
