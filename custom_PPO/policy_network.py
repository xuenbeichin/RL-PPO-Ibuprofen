import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),  # Input layer to hidden layer
            nn.ReLU(),                # Activation function
            nn.Linear(64, action_dim), # Hidden layer to output layer
            nn.Softmax(dim=-1)        # Output probabilities for each action
        )

    def forward(self, state):
        """
        Forward pass through the network.
        """
        return self.fc(state)
