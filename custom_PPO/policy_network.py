import torch.nn as nn

class PolicyNetwork(nn.Module):
    """
    A neural network model that represents the policy in a reinforcement learning agent.

    This class defines the policy network used in Proximal Policy Optimization (PPO) or other RL algorithms.
    It takes the state of the environment as input and outputs a probability distribution over the possible actions
    that the agent can take.

    Attributes:
        fc (nn.Sequential): A sequence of layers that define the neural network architecture.
                             It consists of an input layer, a hidden layer, and an output layer that generates action probabilities.

    Methods:
        forward(state): Performs a forward pass through the network and returns the action probabilities.
    """

    def __init__(self, state_dim, action_dim):
        """
        Initializes the PolicyNetwork.

        Args:
            state_dim (int): The dimensionality of the input state (number of features in the state space).
            action_dim (int): The dimensionality of the output (number of possible actions).

        The network consists of:
            - An input layer with `state_dim` units.
            - A hidden layer with 64 units and a ReLU activation function.
            - An output layer with `action_dim` units, which is passed through a softmax function
              to produce probabilities for each action.
        """
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),  # Input layer to hidden layer (state_dim -> 64 units)
            nn.ReLU(),                 # Activation function (ReLU)
            nn.Linear(64, action_dim), # Hidden layer to output layer (64 -> action_dim)
            nn.Softmax(dim=-1)         # Softmax to normalize the output into probabilities
        )

    def forward(self, state):
        """
        Perform a forward pass through the policy network.

        Args:
            state (torch.Tensor): A tensor representing the current state of the environment.
                                  It should be of shape (batch_size, state_dim).

        Returns:
            torch.Tensor: A tensor representing the action probabilities. The shape is (batch_size, action_dim).
        """
        return self.fc(state)
