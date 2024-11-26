import torch
import torch.nn as nn
from torch import optim
from custom_PPO.policy_network import PolicyNetwork
from custom_PPO.value_network import ValueNetwork

class PPOAgent:
    """
    A Proximal Policy Optimization (PPO) agent with policy and value networks.

    The PPOAgent uses two neural networks:
    - A **policy network** to predict action probabilities given a state.
    - A **value network** to estimate the value of the given state.

    The agent is trained using the PPO algorithm, which aims to optimize the policy while ensuring that the updates are not too large (via the clipped surrogate objective).

    Attributes:
        policy (PolicyNetwork): The policy network, which outputs action probabilities.
        value (ValueNetwork): The value network, which estimates the value of a state.
        optimizer_policy (optim.Adam): The optimizer used to update the policy network.
        optimizer_value (optim.Adam): The optimizer used to update the value network.
        gamma (float): The discount factor used in reward calculation.
        eps_clip (float): The clipping parameter for PPO's surrogate objective to control the magnitude of policy updates.
    """

    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, eps_clip=0.2):
        """
        Initializes the PPO agent with policy and value networks.

        Args:
            state_dim (int): The dimensionality of the state space (number of state features).
            action_dim (int): The dimensionality of the action space (number of possible actions).
            lr (float, optional): The learning rate for the optimizers (default is 0.001).
            gamma (float, optional): The discount factor for future rewards (default is 0.99).
            eps_clip (float, optional): The clipping parameter for PPO's surrogate objective (default is 0.2).
        """
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

        The advantage is computed as the difference between the discounted cumulative return (G_t)
        and the estimated value for the state (V_t). The advantage is used to scale the objective function
        during policy optimization.

        Args:
            rewards (list): A list of rewards observed in the trajectory.
            values (list): A list of state values predicted by the value network.
            dones (list): A list of done flags (whether the episode has ended at each step).

        Returns:
            list: The computed advantages for each state-action pair in the trajectory.
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

        This method performs multiple updates on the policy network using the PPO algorithm
        and updates the value network using the mean squared error (MSE) loss function.

        Args:
            states (list): A list of states encountered during the trajectory.
            actions (list): A list of actions taken by the agent in the trajectory.
            rewards (list): A list of rewards observed in the trajectory.
            dones (list): A list of done flags indicating whether each step was terminal.
            old_probs (list): A list of old action probabilities, used for importance sampling in PPO.
        """
        # Convert data to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        old_probs = torch.tensor(old_probs, dtype=torch.float32)

        # Compute state values and advantages
        values = self.value(states).squeeze()  # Get the predicted values for each state
        advantages = self.compute_advantage(rewards, values.detach().numpy(), dones.numpy())
        advantages = torch.tensor(advantages, dtype=torch.float32)

        # Perform multiple PPO updates
        for _ in range(5):  # PPO multiple updates (usually 5-10)
            # Compute new probabilities for the actions taken
            new_probs = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze()
            ratio = (new_probs / old_probs)  # Importance sampling ratio
            clip = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)  # Clipping to prevent large updates
            policy_loss = -torch.min(ratio * advantages, clip * advantages).mean()  # PPO policy loss

            # Update policy network
            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()

        # Update value network using MSE loss
        value_loss = nn.MSELoss()(self.value(states).squeeze(), rewards)  # MSE loss for value network
        self.optimizer_value.zero_grad()
        value_loss.backward()
        self.optimizer_value.step()
