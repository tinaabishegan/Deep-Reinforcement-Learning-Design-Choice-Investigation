# networks/dqn_network.py

import torch
import torch.nn as nn
from architectures.DoubleDQNPERDuelingNoisy.noisy_linear import NoisyLinear

class DQNNetwork(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        hidden_layers=[64, 64],
        activation_fn=None,  # Expect nn.Module for activation
        init_fn=None,
        sigma_init=0.5,
        seed=42,
        dueling=True
    ):
        super(DQNNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.activation_fn = activation_fn or nn.ReLU()  # Default to nn.ReLU
        self.dueling = dueling

        # Shared feature layers
        layers = []
        input_size = state_size
        for hidden_size in hidden_layers:
            layers.append(NoisyLinear(input_size, hidden_size, sigma_init=sigma_init))
            layers.append(self.activation_fn)  # Add nn.Module, not function
            input_size = hidden_size

        self.feature_layer = nn.Sequential(*layers)

        if dueling:
            # Separate streams for Value and Advantage
            self.value_layer = NoisyLinear(input_size, 1, sigma_init=sigma_init)
            self.advantage_layer = NoisyLinear(input_size, action_size, sigma_init=sigma_init)
        else:
            # Single output layer
            self.output_layer = NoisyLinear(input_size, action_size, sigma_init=sigma_init)

        # Initialize weights if provided
        if init_fn is not None:
            self.apply(init_fn)

    def forward(self, state, noise_scale=None):
        x = state
        for layer in self.feature_layer:
            if isinstance(layer, NoisyLinear):
                x = layer(x, noise_scale=noise_scale)
            else:
                x = layer(x)
        if self.dueling:
            value = self.value_layer(x, noise_scale=noise_scale)
            advantage = self.advantage_layer(x, noise_scale=noise_scale)
            q = value + (advantage - advantage.mean(dim=1, keepdim=True))
            return q
        else:
            x = self.output_layer(x, noise_scale=noise_scale)
            return x

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
