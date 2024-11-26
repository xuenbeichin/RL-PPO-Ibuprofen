import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from networks import PolicyNetwork, ValueNetwork

class PPOAgent:
    """
    Implements the Proximal Policy Optimization (PPO) agent.

    This agent uses separate policy and value networks for action selection and
    value estimation. It employs the PPO algorithm with clipped surrogate loss
    for policy updates and Mean Squared Error (MSE) for value updates.

    Attributes:
        policy (PolicyNetwork): The policy network (actor).
        value (ValueNetwork): The value network (critic).
        optimizer_policy (torch.optim.Adam): Optimizer for the policy network.
        optimizer_value (torch.optim.Adam): Optimizer for the value network.
        gamma (float): Discount factor for rewards.
        eps_clip (float): Clipping range for PPO's surrogate objective.
    """
    def __init__(self, state_dim, action_dim, lr=0.0001, gamma=0.95, eps_clip=0.15):
        """
        Initializes the PPO agent with given hyperparameters.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space (number of possible actions).
            lr (float): Learning rate for both policy and value networks.
            gamma (float): Discount factor for future rewards.
            eps_clip (float): Clipping range for PPO's surrogate objective.
        """
        self.policy = PolicyNetwork(state_dim, action_dim)  # Initialize the policy network.
        self.value = ValueNetwork(state_dim)  # Initialize the value network.
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr)  # Optimizer for policy network.
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=lr)  # Optimizer for value network.
        self.gamma = gamma  # Discount factor for reward calculation.
        self.eps_clip = eps_clip  # Clipping range for PPO.

    def compute_advantage(self, rewards, values, dones):
        """
        Computes the advantage function using Generalized Advantage Estimation (GAE).

        Args:
            rewards (list): Rewards collected during the trajectory.
            values (list): Value function estimates from the value network.
            dones (list): Boolean flags indicating episode termination.

        Returns:
            list: Advantage values for each state-action pair.
        """
        advantage = []  # Store computed advantages.
        g_t = 0  # Initialize the return (discounted cumulative reward).

        # Compute advantage in reverse order for efficiency.
        for reward, value, done in zip(reversed(rewards), reversed(values), reversed(dones)):
            g_t = reward + self.gamma * g_t * (1 - done)  # Update return considering termination.
            advantage.insert(0, g_t - value)  # Advantage: G_t - V(s).

        return advantage

    def train(self, states, actions, rewards, dones, old_probs):
        """
        Performs a training step for both the policy and value networks.

        Args:
            states (list): List of observed states.
            actions (list): List of actions taken.
            rewards (list): List of rewards received.
            dones (list): List of episode termination flags.
            old_probs (list): Action probabilities from the policy at the time of sampling.
        """
        # Convert inputs to PyTorch tensors for computation.
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.int64)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)
        old_probs = torch.tensor(np.array(old_probs), dtype=torch.float32)

        # Compute value estimates and advantages.
        values = self.value(states).squeeze()
        advantages = self.compute_advantage(rewards, values.detach().numpy(), dones.numpy())
        advantages = torch.tensor(advantages, dtype=torch.float32)

        # Update policy network using PPO's clipped surrogate objective.
        for _ in range(5):  # Perform multiple updates for policy stability.
            new_probs = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze()  # Get probabilities for taken actions.
            ratio = new_probs / old_probs  # Importance sampling ratio.
            clip = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)  # Clip the ratio to prevent large updates.

            # Compute policy loss as the minimum of clipped and unclipped objectives.
            policy_loss = -torch.min(ratio * advantages, clip * advantages).mean()

            # Optimize the policy network.
            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()

        # Update value network using Mean Squared Error (MSE) loss.
        value_loss = nn.MSELoss()(self.value(states).squeeze(), rewards)  # Target is reward as return.
        self.optimizer_value.zero_grad()
        value_loss.backward()
        self.optimizer_value.step()
