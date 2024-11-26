import torch.nn as nn

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),  # Input layer to hidden layer
            nn.ReLU(),                # Activation function
            nn.Linear(64, 1)          # Output a single value (state value)
        )

    def forward(self, state):
        """
        Forward pass through the network.
        """
        return self.fc(state)
