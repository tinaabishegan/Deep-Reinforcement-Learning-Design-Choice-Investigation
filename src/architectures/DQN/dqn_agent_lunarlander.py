# architectures/DQN/agent_dqn_lunarlander.py

import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from architectures.DQN.dqn_network import DQNNetwork
from shared.replay_buffer import ReplayMemory
from shared.abstract_policy import AbstractPolicy


class DQNAgentLunarLander(AbstractPolicy):
    def __init__(self, state_size, action_size, hyperparameters, seed=42):
        """
        Initialize the DQN Agent for LunarLander with support for sub-goals.
        """
        # Adjust state size to include sub-goal size
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Hyperparameters
        self.batch_size = hyperparameters["BATCH_SIZE"]
        self.gamma = hyperparameters["GAMMA"]
        self.epsilon_start = hyperparameters["EPSILON_START"]
        self.epsilon_end = hyperparameters["EPSILON_END"]
        self.epsilon_decay = hyperparameters["EPSILON_DECAY"]
        self.epsilon = self.epsilon_start
        self.memory_capacity = hyperparameters["MEMORY_CAPACITY"]
        self.target_update_every = hyperparameters["TARGET_UPDATE_EVERY"]
        self.tau = hyperparameters.get("TAU", 1e-3)

        # Network architecture parameters
        self.hidden_layers = hyperparameters.get("HIDDEN_LAYERS", [64, 64])
        self.activation_fn = hyperparameters.get("ACTIVATION_FN", F.relu)
        self.init_fn = hyperparameters.get("INIT_FN", None)

        # Q-Network
        self.qnetwork_local = DQNNetwork(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_layers=self.hidden_layers,
            activation_fn=self.activation_fn,
            init_fn=self.init_fn,
            seed=seed
        )
        self.qnetwork_target = DQNNetwork(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_layers=self.hidden_layers,
            activation_fn=self.activation_fn,
            init_fn=self.init_fn,
            seed=seed
        )
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=hyperparameters["LEARNING_RATE"])

        # Replay memory
        self.memory = ReplayMemory(self.memory_capacity, self.batch_size, seed)
        self.t_step = 0
        self.last_loss = None

    def act(self, state):
        """
        Select an action using epsilon-greedy policy
        """
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()

        if random.random() < self.epsilon:
            return random.choice(np.arange(self.action_size))
        else:
            return np.argmax(action_values.cpu().data.numpy())

    def step(self, state, action, reward, next_state, done):
        """
        Save experience in replay memory and learn periodically.
        Returns the loss value for TensorBoard logging.
        """

        self.memory.push(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        loss = None
        self.t_step = (self.t_step + 1) % self.target_update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            loss = self.learn(experiences)
            self.last_loss = loss  # Store last loss 
        else:
            self.last_loss = None
        return loss

    def learn(self, experiences):
        """
        Learn from the sampled transitions.
        Returns the loss value for TensorBoard logging.
        """
        states, actions, rewards, next_states, dones = experiences

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Get max predicted Q values from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Compute loss
        loss = F.smooth_l1_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

        return loss.item()

    def soft_update(self, local_model, target_model, tau):
        """
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def update_epsilon(self):
        """
        Decay epsilon after each episode.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
