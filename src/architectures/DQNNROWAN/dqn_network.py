import torch
import torch.nn as nn
import torch.nn.functional as F
from architectures.DQNNROWAN.noisy_linear import NoisyLinear

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

    def forward(self, state):
        x = state
        for layer in self.layers[:-1]:
            x = self.activation_fn(layer(x))
        x = self.layers[-1](x)
        return x

    def reset_noise(self):
        if self.use_noisy:
            for layer in self.layers:
                if isinstance(layer, NoisyLinear):
                    layer.reset_noise()

    def compute_noise_level_D(self):
        """
        Compute the noise level D for the output layer.
        """
        if not self.use_noisy:
            return 0.0

        output_layer = self.layers[-1]
        if isinstance(output_layer, NoisyLinear):
            sigma_w = output_layer.get_sigma_weights()
            sigma_b = output_layer.get_sigma_biases()
            p_star = sigma_w.size(1)  # Input dimension of the last layer
            N_a = sigma_w.size(0)     # Number of actions (output neurons)

            D = (sigma_w.sum() + sigma_b.sum()) / ((p_star + 1) * N_a)
            return D
        else:
            return 0.0  # No noise in output layer
