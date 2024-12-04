import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
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
        self.state_buffer = []
        self.normalize = normalize

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
                penalty = (self.therapeutic_range[0] - self.plasma_concentration) * 0.1
                reward = -5 - penalty
            elif self.plasma_concentration > self.therapeutic_range[1]:
                penalty = (self.plasma_concentration - self.therapeutic_range[1]) * 0.1
                reward = -5 - penalty

        if len(self.state_buffer) > 1:
            fluctuation_penalty = abs(self.state_buffer[-1] - self.state_buffer[-2]) * 0.05
            reward -= fluctuation_penalty

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


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units, num_layers):
        super(PolicyNetwork, self).__init__()
        layers = [nn.Linear(state_dim, hidden_units), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_units, hidden_units), nn.ReLU()]
        layers += [nn.Linear(hidden_units, action_dim), nn.Softmax(dim=-1)]
        self.fc = nn.Sequential(*layers)

    def forward(self, state):
        return self.fc(state)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_units, num_layers):
        super(ValueNetwork, self).__init__()
        layers = [nn.Linear(state_dim, hidden_units), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_units, hidden_units), nn.ReLU()]
        layers.append(nn.Linear(hidden_units, 1))
        self.fc = nn.Sequential(*layers)

    def forward(self, state):
        return self.fc(state)


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, eps_clip, batch_size, ppo_epochs, entropy_beta, buffer_size,
                 max_steps, hidden_units, num_layers, lambda_gae=0.95):
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_units, num_layers)
        self.value = ValueNetwork(state_dim, hidden_units, num_layers)
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.batch_size = batch_size
        self.ppo_epochs = ppo_epochs
        self.entropy_beta = entropy_beta
        self.buffer_size = buffer_size
        self.max_steps = max_steps
        self.lambda_gae = lambda_gae

    def compute_advantage(self, rewards, values, dones):
        advantages = []
        deltas = rewards[:-1] + self.gamma * values[1:] * (1 - dones[1:]) - values[:-1]
        advantage = 0
        for delta, done in zip(reversed(deltas), reversed(dones[:-1])):
            advantage = delta + self.gamma * self.lambda_gae * advantage * (1 - done)
            advantages.insert(0, advantage)

        last_value = 0 if dones[-1] else values[-1]
        advantages.append(last_value)

        return np.array(advantages)

    def train(self, trajectories, step):
        states, actions, rewards, dones, old_probs = map(np.array, trajectories)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        old_probs = torch.tensor(old_probs, dtype=torch.float32)

        # Compute value estimates and advantages
        values = self.value(states).squeeze()
        advantages = self.compute_advantage(rewards.numpy(), values.detach().numpy(), dones.numpy())
        advantages = torch.tensor(advantages, dtype=torch.float32)

        # Normalize advantages
        returns = advantages + values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Initialize accumulators for metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_kl_divergence = 0
        total_clipfrac = 0

        for _ in range(self.ppo_epochs):
            for i in range(0, len(states), self.batch_size):
                # Sample mini-batches
                batch = slice(i, i + self.batch_size)
                state_batch = states[batch]
                action_batch = actions[batch]
                advantage_batch = advantages[batch]
                old_prob_batch = old_probs[batch]
                return_batch = returns[batch]

                # Compute new probabilities
                new_probs = self.policy(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()
                ratio = new_probs / old_prob_batch

                # Policy loss: Clipped surrogate objective
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                # Clip fraction
                clip_fraction = (torch.abs(ratio - 1.0) > self.eps_clip).float().mean().item()
                total_clipfrac += clip_fraction

                # KL divergence
                kl_divergence = (
                        old_prob_batch * (torch.log(old_prob_batch + 1e-10) - torch.log(new_probs + 1e-10))
                ).mean()

                # Entropy loss: Encourage exploration
                entropy_loss = -torch.sum(new_probs * torch.log(new_probs + 1e-10), dim=-1).mean()

                # Accumulate metrics
                total_policy_loss += policy_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_kl_divergence += kl_divergence.item()

                # Final policy loss (includes entropy term)
                policy_loss -= self.entropy_beta * entropy_loss
                self.optimizer_policy.zero_grad()
                policy_loss.backward()
                self.optimizer_policy.step()

                # Value loss: MSE between returns and value predictions
                value_preds = self.value(state_batch).squeeze()
                value_loss = F.mse_loss(value_preds, return_batch)
                total_value_loss += value_loss.item()

                self.optimizer_value.zero_grad()
                value_loss.backward()
                self.optimizer_value.step()

        # Compute averages for logging
        avg_policy_loss = total_policy_loss / (self.ppo_epochs * len(states) / self.batch_size)
        avg_value_loss = total_value_loss / (self.ppo_epochs * len(states) / self.batch_size)
        avg_entropy_loss = total_entropy_loss / (self.ppo_epochs * len(states) / self.batch_size)
        avg_kl_divergence = total_kl_divergence / (self.ppo_epochs * len(states) / self.batch_size)
        avg_clipfrac = total_clipfrac / (self.ppo_epochs * len(states) / self.batch_size)
        total_loss = avg_policy_loss + avg_value_loss - self.entropy_beta * avg_entropy_loss

        # Explained variance
        explained_variance = 1 - torch.var(returns - values).item() / (torch.var(returns).item() + 1e-8)

        # Log metrics to TensorBoard
        writer.add_scalar("loss/policy_gradient_loss", avg_policy_loss, step)
        writer.add_scalar("loss/value_function_loss", avg_value_loss, step)
        writer.add_scalar("loss/entropy_loss", avg_entropy_loss, step)
        writer.add_scalar("loss/approximate_kullback_leibler", avg_kl_divergence, step)
        writer.add_scalar("loss/loss", total_loss, step)
        writer.add_scalar("metrics/clipfrac", avg_clipfrac, step)
        writer.add_scalar("metrics/policy_entropy", avg_entropy_loss, step)
        writer.add_scalar("metrics/explained_variance", explained_variance, step)


# Initialize TensorBoard writer
writer = SummaryWriter(log_dir="runs2/ppo_training")


def objective(trial):
    # Define the search space for time_horizon
    time_horizon = trial.suggest_int("time_horizon", 6, 24, step=2)

    # Other hyperparameters to optimize
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.99)
    eps_clip = trial.suggest_float("eps_clip", 0.1, 0.3)
    ppo_epochs = trial.suggest_int("ppo_epochs", 3, 10)
    entropy_beta = trial.suggest_float("entropy_beta", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 32, 512, step=32)
    buffer_size = trial.suggest_int("buffer_size", batch_size * 10, batch_size * 20, step=batch_size)
    hidden_units = trial.suggest_int("hidden_units", 32, 512, step=32)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    normalize = trial.suggest_categorical("normalize", [True, False])

    # Initialize environment and PPO agent
    env = IbuprofenEnv(normalize=normalize)
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=lr,
        gamma=gamma,
        eps_clip=eps_clip,
        batch_size=batch_size,
        ppo_epochs=ppo_epochs,
        max_steps=env.max_steps,
        entropy_beta=entropy_beta,
        buffer_size=buffer_size,
        hidden_units=hidden_units,
        num_layers=num_layers,
    )

    # Training loop for Optuna
    reward_history = []
    step = 0  # Add a step counter

    for episode in range(100):  # Number of episodes per trial
        states, actions, rewards, dones, old_probs = [], [], [], [], []
        state, _ = env.reset()
        total_reward = 0

        for t in range(time_horizon):  # Use the sampled `time_horizon`
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_probs = agent.policy(state_tensor).detach().numpy()
            action = np.random.choice(env.action_space.n, p=action_probs)

            new_state, reward, done, truncated, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done or truncated)
            old_probs.append(action_probs[action])

            state = new_state
            total_reward += reward

            if done or truncated:
                break

        # Train PPO with collected data
        agent.train((states, actions, rewards, dones, old_probs), step=step)
        step += 1  # Increment step counter for each training call
        reward_history.append(total_reward)

    # Return mean reward over episodes for Optuna to optimize
    return np.mean(reward_history)


# Run the Optuna Optimization
env = IbuprofenEnv(normalize=True)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# Print the Best Hyperparameters
print("Best Hyperparameters:")
print(study.best_params)

# Train the Agent with the Best Hyperparameters
best_params = study.best_params
agent = PPOAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    lr=best_params["learning_rate"],
    gamma=best_params["gamma"],
    eps_clip=best_params["eps_clip"],
    batch_size=best_params["batch_size"],
    ppo_epochs=best_params["ppo_epochs"],
    entropy_beta=best_params["entropy_beta"],
    buffer_size=best_params["buffer_size"],
    max_steps=env.max_steps,
    hidden_units=best_params["hidden_units"],
    num_layers=best_params["num_layers"],
)


batch_size = best_params["batch_size"]

# Initialize variables for dynamic time horizon
initial_horizon = 6  # Start with a small time horizon
max_horizon = 24     # Full time period (24 hours, in your case)
horizon_increment = 2  # Increase the horizon incrementally
time_horizon = initial_horizon


# Setup TensorBoard writer
writer = SummaryWriter(log_dir="runs")

# Training loop
episodes = 1000  # Total training episodes
reward_history = []
step = 0  # Initialize the step counter

# Main training loop
for episode in range(episodes):
    states, actions, rewards, dones, old_probs = [], [], [], [], []
    state, _ = env.reset()
    total_reward = 0

    for t in range(env.max_steps):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probs = agent.policy(state_tensor).detach().numpy().flatten()
        action = np.random.choice(action_dim, p=action_probs)

        next_state, reward, done, truncated, _ = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done or truncated)
        old_probs.append(action_probs[action])

        state = next_state
        total_reward += reward

        if done or truncated:
            break

    # Train PPO agent with collected trajectories
    agent.train((states, actions, rewards, dones, old_probs), step=step)
    step += 1  # Increment the step counter

    # Log reward and metrics at the episode level
    writer.add_scalar("reward/total_reward", total_reward, episode)
    print(f"Episode {episode}/{episodes} - Total Reward: {total_reward:.2f}")



# Plot training rewards
plt.figure(figsize=(12, 6))
plt.plot(reward_history, label="Total Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Learning Curve (Custom PPO)")
plt.legend()
plt.grid()
plt.show()

# Finalize TensorBoard
writer.close()


# Evaluation Loop
evaluation_episodes = 100  # Number of episodes for evaluation
state, _ = env.reset()

evaluation_rewards = []
plasma_concentration_trajectories = []

for episode in range(evaluation_episodes):
    state, _ = env.reset()
    total_reward = 0
    plasma_concentration_history = [state[0]]  # Track plasma concentration

    for _ in range(env.max_steps):  # Use max_steps from environment
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probs = agent.policy(state_tensor).detach().numpy().flatten()
        action = np.argmax(action_probs)  # Greedy action selection for evaluation

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
plt.title("Plasma Concentration Over Time (Custom PPO)")
plt.legend()
plt.grid()
plt.show()


