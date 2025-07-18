# networks/dqn_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        hidden_layers=[64, 64],
        activation_fn=F.relu,
        init_fn=None,
        seed=42
    ):
        super(DQNNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.activation_fn = activation_fn

        # Create a list to hold the layers
        layers = []

        # Input layer
        input_size = state_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size

        # Output layer
        layers.append(nn.Linear(input_size, action_size))

        # Assign layers as a ModuleList so that they're registered properly
        self.layers = nn.ModuleList(layers)

        # Initialize weights if an initialization function is provided
        if init_fn is not None:
            self.apply(init_fn)

    def forward(self, state):
        x = state
        for layer in self.layers[:-1]:
            x = self.activation_fn(layer(x))
        # No activation on the output layer
        x = self.layers[-1](x)
        return x
