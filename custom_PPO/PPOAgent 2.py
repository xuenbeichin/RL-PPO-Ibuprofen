import numpy as np
import torch
import torch.nn as nn
from torch import optim


from custom_PPO.networks import PolicyNetwork, ValueNetwork


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
        self.lambda_gae = lambda_gae  # For GAE

    def compute_advantage(self, rewards, values, dones):
        """
        Compute advantages and discounted rewards in a fully vectorized manner using PyTorch.
        """
        # Convert everything to PyTorch tensors
        rewards = torch.tensor(rewards, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Append bootstrap value for the last state
        values = torch.cat([values, torch.tensor([0.0] if dones[-1] else [values[-1]])])

        # Compute deltas
        deltas = rewards + self.gamma * values[1:] * (1 - dones) - values[:-1]

        # Compute advantages using reverse cumulative sum
        advantages = torch.zeros_like(deltas)
        discount = self.gamma * self.lambda_gae
        for t in reversed(range(len(deltas))):
            advantages[t] = deltas[t] + (discount * advantages[t + 1] if t + 1 < len(deltas) else 0)

        return advantages.numpy()  # Convert back to NumPy for compatibility if needed

    def train(self, trajectories):
        states, actions, rewards, dones, old_probs = map(np.array, trajectories)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        old_probs = torch.tensor(old_probs, dtype=torch.float32)

        # Compute returns
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            G = reward + (self.gamma * G * (1 - done))
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize rewards for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute value loss and optimize the value network
        predicted_values = self.value(states).squeeze()
        value_loss = nn.MSELoss()(predicted_values, returns)
        self.optimizer_value.zero_grad()
        value_loss.backward()
        self.optimizer_value.step()

        # Compute advantages
        advantages = returns - predicted_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Optimize the policy network
        for _ in range(self.ppo_epochs):
            for i in range(0, len(states), self.batch_size):
                batch = slice(i, i + self.batch_size)
                state_batch, action_batch, advantage_batch, old_prob_batch = (
                    states[batch], actions[batch], advantages[batch], old_probs[batch])

                action_probs = self.policy(state_batch)
                new_probs = action_probs.gather(1, action_batch.unsqueeze(1)).squeeze()
                ratio = new_probs / old_prob_batch
                clip = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
                policy_loss = -torch.min(ratio * advantage_batch, clip * advantage_batch).mean()
                entropy_loss = -torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=-1).mean()
                policy_loss -= self.entropy_beta * entropy_loss

                self.optimizer_policy.zero_grad()
                policy_loss.backward()
                self.optimizer_policy.step()
                self.optimizer_policy.step()