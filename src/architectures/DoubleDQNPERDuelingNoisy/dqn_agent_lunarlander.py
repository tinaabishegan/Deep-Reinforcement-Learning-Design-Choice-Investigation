import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from architectures.DoubleDQNPERDuelingNoisy.noisy_linear import NoisyLinear
from architectures.DoubleDQNPERDuelingNoisy.dqn_network import DQNNetwork
from shared.replay_buffer import PrioritizedReplayMemory
from shared.abstract_policy import AbstractPolicy

class DQNAgentLunarLander(AbstractPolicy):
    def __init__(self, state_size, action_size, hyperparameters, seed=42, writer=None):
        """
        HRL-enhanced Double DQN with PER, Dueling Network, and Noisy Layers for LunarLander.
        """
        self.state_size = state_size
        self.action_size = action_size

        # TensorBoard writer
        self.writer = writer

        # Hyperparameters
        self.batch_size = hyperparameters["BATCH_SIZE"]
        self.gamma = hyperparameters["GAMMA"]
        self.epsilon_start = hyperparameters["EPSILON_START"]
        self.epsilon_end = hyperparameters["EPSILON_END"]
        self.epsilon_decay = hyperparameters["EPSILON_DECAY"]
        self.epsilon = self.epsilon_start
        self.target_update_every = hyperparameters["TARGET_UPDATE_EVERY"]
        self.tau = hyperparameters.get("TAU", 1e-3)
        self.sigma_init = hyperparameters.get("SIGMA_INIT", 0.5)

        # Replay memory with PER
        self.memory = PrioritizedReplayMemory(
            hyperparameters["MEMORY_CAPACITY"],
            self.batch_size,
            seed,
            alpha=hyperparameters.get("PER_ALPHA", 0.6),
            beta_start=hyperparameters.get("PER_BETA_START", 0.4),
            beta_frames=hyperparameters.get("PER_BETA_FRAMES", 100000),
        )

        # Initialize networks with dueling architecture
        self.qnetwork_local = DQNNetwork(
            state_size,
            action_size,
            hidden_layers=hyperparameters.get("HIDDEN_LAYERS", [128, 128]),
            activation_fn=hyperparameters.get("ACTIVATION_FN", nn.ReLU()),
            init_fn=hyperparameters.get("INIT_FN", None),
            sigma_init=self.sigma_init,
            dueling=True,
        )
        self.qnetwork_target = DQNNetwork(
            state_size,
            action_size,
            hidden_layers=hyperparameters.get("HIDDEN_LAYERS", [128, 128]),
            activation_fn=hyperparameters.get("ACTIVATION_FN", nn.ReLU()),
            init_fn=hyperparameters.get("INIT_FN", None),
            sigma_init=self.sigma_init,
            dueling=True,
        )

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=hyperparameters["LEARNING_RATE"])

        # Learning step counter
        self.learn_step_counter = 0
        self.t_step = 0
        self.last_loss = None


    def act(self, state, sub_goal=None):
        """
        HRL-compatible action selection using Noisy Layers and epsilon-greedy strategy.
        """
        self.qnetwork_local.reset_noise()
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        if random.random() < self.epsilon:
            return random.choice(np.arange(self.action_size))
        else:
            return np.argmax(action_values.cpu().data.numpy())

    def step(self, state, action, reward, next_state, done):
        """
        Store transition and trigger learning.
        """
        self.memory.push(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY steps
        loss = None
        self.t_step = (self.t_step + 1) % self.target_update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences, indices, weights = self.memory.sample()
                loss = self.learn(experiences, indices, weights)
                self.last_loss = loss  # Store last loss 
                if self.writer:
                    self.writer.add_scalar("Loss/LunarLander", loss, self.learn_step_counter)
        else:
            self.last_loss = None
        return loss

    def learn(self, experiences, indices, weights):
        """
        Double DQN learning with PER and Noisy Layers.
        """
        if self.learn_step_counter % self.target_update_every == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.learn_step_counter += 1

        states, actions, rewards, next_states, dones = experiences

        # Compute Q-values
        q_eval = self.qnetwork_local(states).gather(1, actions)

        # Double DQN with PER
        next_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        q_next = self.qnetwork_target(next_states).gather(1, next_actions).detach()
        q_target = rewards + self.gamma * q_next * (1 - dones)
        td_errors = q_target - q_eval

        # Weighted loss with PER
        weights_tensor = torch.tensor(weights, dtype=torch.float32).to(states.device)
        loss = (weights_tensor * F.smooth_l1_loss(q_eval, q_target, reduction="none")).mean()

        # Update priorities
        abs_td_errors = torch.abs(td_errors).detach().cpu().numpy()
        self.memory.update_priorities(indices, abs_td_errors)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Adjust noise scale in noisy layers
        self.adjust_noise_scale(td_errors)
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        self.qnetwork_local.reset_noise()
        self.qnetwork_target.reset_noise()

        return loss.item()

    def adjust_noise_scale(self, td_errors):
        """
        Adjust noise scale in Noisy Layers based on TD error.
        """
        mean_td_error = td_errors.abs().mean().item()
        new_scale = max(0.01, min(1.0, mean_td_error))
        for module in self.qnetwork_local.modules():
            if isinstance(module, NoisyLinear):
                module.set_noise_scale(new_scale)
        for module in self.qnetwork_target.modules():
            if isinstance(module, NoisyLinear):
                module.set_noise_scale(new_scale)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def update_epsilon(self):
        """
        Decay epsilon for epsilon-greedy exploration.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
