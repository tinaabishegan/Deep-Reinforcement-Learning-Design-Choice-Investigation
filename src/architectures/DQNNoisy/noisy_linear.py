# networks/noisy_linear.py

import torch
import torch.nn as nn
import math

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5, adaptive_noise=True):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        self.adaptive_noise = adaptive_noise

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Buffers for noise
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        # Initialize mu and sigma
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        # Generate new noise
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input, noise_scale=None):
        if noise_scale is not None:
            scale = noise_scale
        else:
            scale = self.noise_scale if hasattr(self, 'noise_scale') else 1.0

        if self.training or self.adaptive_noise:
            weight = self.weight_mu + scale * self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + scale * self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return nn.functional.linear(input, weight, bias)


    @staticmethod
    def _scale_noise(size):
        # Factorized Gaussian noise
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt())
    
    
    def set_noise_scale(self, scale):
        self.noise_scale = scale