import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from architectures.DoubleDQNPERDuelingNROWAN.dqn_network import DQNNetwork
from shared.replay_buffer import PrioritizedReplayMemory
from shared.abstract_policy import AbstractPolicy


class DQNAgentLunarLander(AbstractPolicy):
    def __init__(self, state_size, action_size, hyperparameters, seed=42, writer=None):
        """
        Initialize the Double DQN agent for LunarLander with PER, dueling architecture, and NROWAN.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)

        # TensorBoard writer for logging
        self.writer = writer

        # Hyperparameters
        self.batch_size = hyperparameters["BATCH_SIZE"]
        self.learning_rate = hyperparameters["LEARNING_RATE"]
        self.gamma = hyperparameters["GAMMA"]
        self.epsilon_start = hyperparameters["EPSILON_START"]
        self.epsilon_end = hyperparameters["EPSILON_END"]
        self.epsilon_decay = hyperparameters["EPSILON_DECAY"]
        self.epsilon = self.epsilon_start
        self.memory_capacity = hyperparameters["MEMORY_CAPACITY"]
        self.target_update_every = hyperparameters["TARGET_UPDATE_EVERY"]
        self.tau = hyperparameters.get("TAU", 1e-3)

        # PER parameters
        self.alpha = hyperparameters.get("PER_ALPHA", 0.6)
        self.beta_start = hyperparameters.get("PER_BETA_START", 0.4)
        self.beta_frames = hyperparameters.get("PER_BETA_FRAMES", 100000)

        # Noise-related hyperparameters
        self.sigma_init = hyperparameters.get("SIGMA_INIT", 0.5)

        # Network architecture parameters
        self.hidden_layers = hyperparameters.get("HIDDEN_LAYERS", [128, 128])
        self.activation_fn = hyperparameters.get("ACTIVATION_FN", torch.nn.ReLU())
        self.init_fn = hyperparameters.get("INIT_FN", None)

        # Q-Networks with dueling architecture enabled
        self.qnetwork_local = DQNNetwork(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_layers=self.hidden_layers,
            activation_fn=self.activation_fn,
            init_fn=self.init_fn,
            sigma_init=self.sigma_init,
            dueling=True,
            seed=seed
        )
        self.qnetwork_target = DQNNetwork(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_layers=self.hidden_layers,
            activation_fn=self.activation_fn,
            init_fn=self.init_fn,
            sigma_init=self.sigma_init,
            dueling=True,
            seed=seed
        )
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)

        # Replay memory with Prioritized Experience Replay
        self.memory = PrioritizedReplayMemory(
            self.memory_capacity,
            self.batch_size,
            seed,
            alpha=self.alpha,
            beta_start=self.beta_start,
            beta_frames=self.beta_frames
        )
        self.t_step = 0

        # Noise reduction and weight adjustment
        self.cumulative_reward = 0.0
        self.k_final = hyperparameters.get("K_FINAL", 1.0)
        self.inf_reward = hyperparameters.get("INF_REWARD", -200)
        self.sup_reward = hyperparameters.get("SUP_REWARD", 200)
        self.k = 0.0
        self.last_loss = None

    def act(self, state, sub_goal=None):
        """
        HRL-compatible action selection using NoisyNet exploration.
        """
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.reset_noise()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_size)
        else:
            return np.argmax(action_values.cpu().data.numpy())

    def step(self, state, action, reward, next_state, done):
        """
        Store transitions in replay buffer and update cumulative reward.
        """
        self.memory.push(state, action, reward, next_state, done)
        self.cumulative_reward += reward
        if done:
            self.cumulative_reward = 0.0
        loss = None
        # Perform learning step if enough samples are available
        self.t_step = (self.t_step + 1) % self.target_update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences, indices, weights = self.memory.sample()
            loss = self.learn(experiences, indices, weights)
            self.last_loss = loss  # Store last loss 
            if self.writer:
                self.writer.add_scalar("Loss/LunarLander", loss, self.t_step)
        else:
            self.last_loss = None
        return loss

    def learn(self, experiences, indices, weights):
        """
        Learn from transitions using Double DQN with PER and NROWAN enhancements.
        """
        states, actions, rewards, next_states, dones = experiences

        # Compute Q-values
        q_eval = self.qnetwork_local(states).gather(1, actions)
        next_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        q_next = self.qnetwork_target(next_states).gather(1, next_actions).detach()
        q_target = rewards + self.gamma * q_next * (1 - dones)
        td_errors = q_target - q_eval

        # Weighted loss using PER
        weights_tensor = weights.to(states.device)
        loss = (weights_tensor * F.smooth_l1_loss(q_eval, q_target, reduction="none")).mean()

        # Update priorities in replay buffer
        abs_td_errors = torch.abs(td_errors).detach().cpu().numpy()
        self.memory.update_priorities(indices, abs_td_errors)

        # Noise level D and cumulative reward scaling
        D = self.qnetwork_local.compute_noise_level_D()
        self.k = max(0.0, min(self.k_final, self.k_final * (self.cumulative_reward - self.inf_reward) / (self.sup_reward - self.inf_reward)))
        total_loss = loss + self.k * D

        # Optimize the model
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Soft update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

        return total_loss.item()

    def soft_update(self, local_model, target_model, tau):
        """
        Perform soft update of target network parameters.
        θ_target = τ * θ_local + (1 - τ) * θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def update_epsilon(self):
        """
        Decay epsilon after each episode.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
