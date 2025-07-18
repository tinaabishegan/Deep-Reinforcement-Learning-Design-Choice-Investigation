# networks/dqn_network.py

import torch
import torch.nn as nn
from architectures.DoubleDQNPERDuelingNROWAN.noisy_linear import NoisyLinear

class DQNNetwork(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        hidden_layers=[64, 64],
        activation_fn=nn.ReLU(),
        init_fn=None,
        sigma_init=0.5,
        seed=42,
        dueling=True
    ):
        super(DQNNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.activation_fn = activation_fn
        self.dueling = dueling

        # Shared feature layers
        layers = []
        input_size = state_size
        for hidden_size in hidden_layers:
            layers.append(NoisyLinear(input_size, hidden_size, sigma_init=sigma_init))
            layers.append(self.activation_fn)
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

    def forward(self, state):
        x = state
        x = self.feature_layer(x)
        if self.dueling:
            value = self.value_layer(x)
            advantage = self.advantage_layer(x)
            q = value + (advantage - advantage.mean(dim=1, keepdim=True))
            return q
        else:
            x = self.output_layer(x)
            return x

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def compute_noise_level_D(self):
        """
        Compute the noise level D for the output layer.
        """
        D = 0.0
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                D += module.get_noise_level()
        return D
