import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from architectures.DQNNoisy.noisy_linear import NoisyLinear
from architectures.DQNNoisy.dqn_network import DQNNetwork
from shared.replay_buffer import ReplayMemory
from shared.abstract_policy import AbstractPolicy


class DQNAgentLunarLander(AbstractPolicy):
    def __init__(self, state_size, action_size, hyperparameters, seed=42):
        """
        Initialize the DQN Agent for LunarLander.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Hyperparameters
        self.batch_size = hyperparameters['BATCH_SIZE']
        self.gamma = hyperparameters['GAMMA']
        self.epsilon_start = hyperparameters['EPSILON_START']
        self.epsilon_end = hyperparameters['EPSILON_END']
        self.epsilon_decay = hyperparameters['EPSILON_DECAY']
        self.epsilon = self.epsilon_start
        self.memory_capacity = hyperparameters['MEMORY_CAPACITY']
        self.target_update_every = hyperparameters['TARGET_UPDATE_EVERY']
        self.tau = hyperparameters.get('TAU', 1e-3)
        self.use_noisy = hyperparameters.get('USE_NOISY', False)
        self.sigma_init = hyperparameters.get('SIGMA_INIT', 0.5)

        # Network architecture parameters
        self.hidden_layers = hyperparameters.get('HIDDEN_LAYERS', [128, 128])
        self.activation_fn = hyperparameters.get('ACTIVATION_FN', F.relu)
        self.init_fn = hyperparameters.get('INIT_FN', None)

        # Replay memory
        self.memory = ReplayMemory(self.memory_capacity, self.batch_size, seed)

        # Q-Networks with Noisy layers if enabled
        self.qnetwork_local = DQNNetwork(
            state_size,
            action_size,
            hidden_layers=self.hidden_layers,
            activation_fn=self.activation_fn,
            init_fn=self.init_fn,
            use_noisy=self.use_noisy,
            sigma_init=self.sigma_init,
            seed=seed,
        )
        self.qnetwork_target = DQNNetwork(
            state_size,
            action_size,
            hidden_layers=self.hidden_layers,
            activation_fn=self.activation_fn,
            init_fn=self.init_fn,
            use_noisy=self.use_noisy,
            sigma_init=self.sigma_init,
            seed=seed,
        )
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=hyperparameters['LEARNING_RATE'])

        self.t_step = 0  # Tracks steps for periodic updates
        self.last_loss = None

    def act(self, state, sub_goal=None):
        """
        Hybrid action selection with noisy layers and epsilon-greedy.
        Optionally incorporates sub-goal for HRL compatibility.
        """
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)

        if self.use_noisy:
            self.qnetwork_local.reset_noise()
            with torch.no_grad():
                action_values = self.qnetwork_local(state_tensor)

            if np.random.rand() < self.epsilon:
                return random.choice(np.arange(self.action_size))
            else:
                return np.argmax(action_values.cpu().data.numpy())
        else:
            if np.random.rand() < self.epsilon:
                return random.choice(np.arange(self.action_size))
            else:
                with torch.no_grad():
                    action_values = self.qnetwork_local(state_tensor)
                return np.argmax(action_values.cpu().data.numpy())

    def step(self, state, action, reward, next_state, done):
        """
        Store the transition and learn periodically.
        Returns loss value for TensorBoard logging.
        """
        self.memory.push(state, action, reward, next_state, done)
        loss = None
        self.t_step = (self.t_step + 1) % self.target_update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            loss = self.learn()
            self.last_loss = loss  # Store last loss 
        else:
            self.last_loss = None
        return loss

    def learn(self):
        """
        Learn from replay memory.
        Returns loss value for metrics tracking.
        """
        # Sample transitions from memory
        states, actions, rewards, next_states, dones = self.memory.sample()

        # Get expected Q-values from local model
        q_eval = self.qnetwork_local(states).gather(1, actions)

        # Double DQN: Get next actions from local model, target Q-values from target model
        next_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        q_next = self.qnetwork_target(next_states).gather(1, next_actions).detach()
        q_target = rewards + (self.gamma * q_next * (1 - dones))

        # Compute TD errors and loss
        td_errors = q_target - q_eval
        loss = F.smooth_l1_loss(q_eval, q_target)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Adjust noise scale if noisy layers are enabled
        if self.use_noisy:
            self.adjust_noise_scale(td_errors)
            self.qnetwork_local.reset_noise()
            self.qnetwork_target.reset_noise()

        # Soft update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

        return loss.item()

    def adjust_noise_scale(self, td_errors):
        """
        Adjust noise scale based on TD error magnitude.
        """
        mean_td_error = td_errors.abs().mean().item()
        new_scale = max(0.1, min(1.0, mean_td_error))  # Example scaling logic
        for layer in self.qnetwork_local.layers:
            if isinstance(layer, NoisyLinear):
                layer.set_noise_scale(new_scale)
        for layer in self.qnetwork_target.layers:
            if isinstance(layer, NoisyLinear):
                layer.set_noise_scale(new_scale)

    def soft_update(self, local_model, target_model, tau):
        """
        Perform a soft update of target network parameters.
        θ_target = τ * θ_local + (1 - τ) * θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def update_epsilon(self):
        """
        Decay epsilon after each episode.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
