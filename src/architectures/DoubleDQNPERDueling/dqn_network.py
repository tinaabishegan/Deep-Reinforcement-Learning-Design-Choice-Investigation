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
        dueling=False,  # Toggle for dueling architecture
        seed=42
    ):
        super(DQNNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.activation_fn = activation_fn
        self.dueling = dueling  # Store the dueling flag

        # Feature extraction layers
        layers = []
        input_size = state_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size

        self.feature_layer = nn.Sequential(*layers)

        if self.dueling:
            # Dueling Network Streams
            self.value_layer = nn.Linear(input_size, 1)  # Value stream
            self.advantage_layer = nn.Linear(input_size, action_size)  # Advantage stream
        else:
            # Standard output layer
            self.output_layer = nn.Linear(input_size, action_size)

        # Initialize weights if an initialization function is provided
        if init_fn is not None:
            self.apply(init_fn)

    def forward(self, state):
        # Extract features
        x = self.feature_layer(state)
        x = self.activation_fn(x)

        if self.dueling:
            # Compute Value and Advantage streams
            value = self.value_layer(x)
            advantage = self.advantage_layer(x)

            # Combine to get Q-values
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
            return q_values
        else:
            # Standard Q-values
            q_values = self.output_layer(x)
            return q_values
