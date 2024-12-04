import torch.nn as nn

class PolicyNetwork(nn.Module):
    """
    Neural network representing the policy in PPO.

    The policy outputs a probability distribution over actions given the state.

    Attributes:
        fc (nn.Sequential): A sequential neural network with:
            - An input layer that maps the state to a 64-dimensional space.
            - A ReLU activation.
            - A second layer that maps the 64-dimensional space to action probabilities.
            - A softmax activation to produce a probability distribution.
    """
    def __init__(self, state_dim, action_dim):
        """
        Initializes the PolicyNetwork.

        Args:
            state_dim (int): The dimension of the input state space.
            action_dim (int): The number of possible actions in the action space.
        """
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),  # Input layer: maps state to 64 dimensions.
            nn.ReLU(),                # Activation layer.
            nn.Linear(64, action_dim),  # Output layer: maps 64 dimensions to action probabilities.
            nn.Softmax(dim=-1)        # Softmax to produce a probability distribution.
        )

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Probability distribution over actions.
        """
        return self.fc(state)


class ValueNetwork(nn.Module):
    """
    Neural network representing the value function in PPO.

    The value network estimates the expected cumulative reward (value) for a given state.

    Attributes:
        fc (nn.Sequential): A sequential neural network with:
            - An input layer that maps the state to a 64-dimensional space.
            - A ReLU activation.
            - A second layer that maps the 64-dimensional space to a scalar value.
    """
    def __init__(self, state_dim):
        """
        Initializes the ValueNetwork.

        Args:
            state_dim (int): The dimension of the input state space.
        """
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),  # Input layer: maps state to 64 dimensions.
            nn.ReLU(),                # Activation layer.
            nn.Linear(64, 1)          # Output layer: maps 64 dimensions to a single scalar (state value).
        )

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: The estimated value of the input state.
        """
        return self.fc(state)
