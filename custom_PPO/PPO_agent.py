import torch
import torch.nn as nn
from torch import optim
from PPO.policy_network import PolicyNetwork
from PPO.value_network import ValueNetwork


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, eps_clip=0.2):
        # Initialize policy and value networks
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        # Optimizers for policy and value networks
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=lr)
        # Discount factor and clipping parameter for PPO
        self.gamma = gamma
        self.eps_clip = eps_clip

    def compute_advantage(self, rewards, values, dones):
        """
        Compute advantage estimates based on rewards, values, and done flags.
        """
        advantage = []
        g_t = 0  # Initialize cumulative return
        for reward, value, done in zip(reversed(rewards), reversed(values), reversed(dones)):
            g_t = reward + self.gamma * g_t * (1 - done)  # Discounted future return
            advantage.insert(0, g_t - value)  # Advantage is return minus value
        return advantage

    def train(self, states, actions, rewards, dones, old_probs):
        """
        Train the policy and value networks using collected trajectories.
        """
        # Convert data to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        old_probs = torch.tensor(old_probs, dtype=torch.float32)

        # Compute state values and advantages
        values = self.value(states).squeeze()
        advantages = self.compute_advantage(rewards, values.detach().numpy(), dones.numpy())
        advantages = torch.tensor(advantages, dtype=torch.float32)

        # Perform multiple PPO updates
        for _ in range(5):
            # Compute new probabilities for the actions taken
            new_probs = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze()
            ratio = (new_probs / old_probs)  # Importance sampling ratio
            clip = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
            policy_loss = -torch.min(ratio * advantages, clip * advantages).mean()  # PPO loss

            # Update policy network
            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()

        # Update value network using MSE loss
        value_loss = nn.MSELoss()(self.value(states).squeeze(), rewards)
        self.optimizer_value.zero_grad()
        value_loss.backward()
        self.optimizer_value.step()
