# networks/dqn_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from architectures.DQNNoisy.noisy_linear import NoisyLinear

class DQNNetwork(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        hidden_layers=[64, 64],
        activation_fn=F.relu,
        init_fn=None,
        use_noisy=False,
        sigma_init=0.5,
        seed=42
    ):
        super(DQNNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.activation_fn = activation_fn
        self.use_noisy = use_noisy

        # Create a list to hold the layers
        layers = []

        # Input layer
        input_size = state_size
        for hidden_size in hidden_layers:
            if use_noisy:
                layers.append(NoisyLinear(input_size, hidden_size, sigma_init=sigma_init))
            else:
                layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size

        # Output layer
        if use_noisy:
            layers.append(NoisyLinear(input_size, action_size, sigma_init=sigma_init))
        else:
            layers.append(nn.Linear(input_size, action_size))

        # Assign layers as a ModuleList
        self.layers = nn.ModuleList(layers)

        # Initialize weights if an initialization function is provided
        if init_fn is not None and not use_noisy:
            self.apply(init_fn)

    def forward(self, state, noise_scale=None):
        x = state
        for layer in self.layers[:-1]:
            if isinstance(layer, NoisyLinear):
                x = self.activation_fn(layer(x, noise_scale=noise_scale))
            else:
                x = self.activation_fn(layer(x))
        if isinstance(self.layers[-1], NoisyLinear):
            x = self.layers[-1](x, noise_scale=noise_scale)
        else:
            x = self.layers[-1](x)
        return x


    def reset_noise(self):
        if self.use_noisy:
            for layer in self.layers:
                if isinstance(layer, NoisyLinear):
                    layer.reset_noise()
