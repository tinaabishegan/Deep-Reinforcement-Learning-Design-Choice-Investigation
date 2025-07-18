import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from architectures.DQNNROWAN.noisy_linear import NoisyLinear
from architectures.DQNNROWAN.dqn_network import DQNNetwork
from shared.replay_buffer import ReplayMemory
from shared.abstract_policy import AbstractPolicy


class DQNAgentLunarLander(AbstractPolicy):
    def __init__(self, state_size, action_size, hyperparameters, seed=42, writer=None):
        """
        Initialize the DQNNROWAN agent for LunarLander.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # TensorBoard writer
        self.writer = writer

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

        # Replay memory
        self.memory = ReplayMemory(self.memory_capacity, self.batch_size, seed)

        # Q-Networks
        self.qnetwork_local = DQNNetwork(
            state_size,
            action_size,
            hidden_layers=hyperparameters.get("HIDDEN_LAYERS", [128, 128]),
            activation_fn=hyperparameters.get("ACTIVATION_FN", F.relu),
            init_fn=hyperparameters.get("INIT_FN", None),
            use_noisy=self.use_noisy,
            sigma_init=self.sigma_init,
            seed=seed,
        )
        self.qnetwork_target = DQNNetwork(
            state_size,
            action_size,
            hidden_layers=hyperparameters.get("HIDDEN_LAYERS", [128, 128]),
            activation_fn=hyperparameters.get("ACTIVATION_FN", F.relu),
            init_fn=hyperparameters.get("INIT_FN", None),
            use_noisy=self.use_noisy,
            sigma_init=self.sigma_init,
            seed=seed,
        )
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=hyperparameters['LEARNING_RATE'])
        self.t_step = 0

        # Noise-related parameters
        self.cumulative_reward = 0.0
        self.k_final = hyperparameters.get('K_FINAL', 1.0)
        self.inf_reward = hyperparameters.get('INF_REWARD', -200)
        self.sup_reward = hyperparameters.get('SUP_REWARD', 200)
        self.k = 0.0
        self.last_loss = None

    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        self.cumulative_reward += reward

        if done:
            self.cumulative_reward = 0.0

        self.t_step = (self.t_step + 1) % self.target_update_every
        loss = None
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            loss = self.learn()
            self.last_loss = loss  # Store last loss 
            if self.writer:
                self.writer.add_scalar('Loss/LunarLander', loss, self.t_step)
        else:
            self.last_loss = None
        return loss

    def act(self, state, sub_goal=None):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        if self.use_noisy:
            self.qnetwork_local.reset_noise()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        if random.random() < self.epsilon:
            return random.choice(np.arange(self.action_size))
        else:
            return np.argmax(action_values.cpu().data.numpy())
        # return np.argmax(action_values.cpu().data.numpy())

    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample()

        q_eval = self.qnetwork_local(states).gather(1, actions)
        next_actions = self.qnetwork_local(next_states).detach().max(1)[0].unsqueeze(1)
        q_next = self.qnetwork_target(next_states).gather(1, next_actions).detach()
        q_target = rewards + self.gamma * q_next * (1 - dones)

        td_errors = q_target - q_eval
        loss = F.smooth_l1_loss(q_eval, q_target)

        D = self.qnetwork_local.compute_noise_level_D()

        self.k = self.k_final * (self.cumulative_reward - self.inf_reward) / (self.sup_reward - self.inf_reward)
        self.k = max(0.0, min(self.k_final, self.k))

        total_loss = loss + self.k * D
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        if self.use_noisy:
            self.qnetwork_local.reset_noise()
            self.qnetwork_target.reset_noise()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

        return total_loss.item()

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def update_epsilon(self):
        """
        Decay epsilon after each episode.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)