import torch
import torch.nn.functional as F
import numpy as np
from architectures.DoubleDQNPERDuelingNROWAN.dqn_network import DQNNetwork
from shared.replay_buffer import PrioritizedReplayMemory
from shared.abstract_policy import AbstractPolicy


class DQNAgentCartPole(AbstractPolicy):
    def __init__(self, state_size, action_size, hyperparameters, writer=None):
        """
        Initialize the Double DQN agent with PER, dueling architecture, and NROWAN enhancements.
        """
        # Environment configuration
        self.num_states = state_size
        self.num_actions = action_size

        # TensorBoard writer
        self.writer = writer

        # Hyperparameters
        self.batch_size = hyperparameters["BATCH_SIZE"]
        self.learning_rate = hyperparameters["LEARNING_RATE"]
        self.gamma = hyperparameters["GAMMA"]
        self.epsilon = hyperparameters["EPSILON_START"]
        self.min_epsilon = hyperparameters["EPSILON_END"]
        self.epsilon_decay = hyperparameters["EPSILON_DECAY"]
        self.target_update_iter = hyperparameters["TARGET_UPDATE_EVERY"]

        # PER parameters
        self.alpha = hyperparameters.get("PER_ALPHA", 0.6)
        self.beta_start = hyperparameters.get("PER_BETA_START", 0.4)
        self.beta_frames = hyperparameters.get("PER_BETA_FRAMES", 100000)

        # Replay memory
        self.memory = PrioritizedReplayMemory(
            hyperparameters["MEMORY_CAPACITY"],
            self.batch_size,
            seed=42,
            alpha=self.alpha,
            beta_start=self.beta_start,
            beta_frames=self.beta_frames,
        )

        # Initialize networks
        self.sigma_init = hyperparameters.get("SIGMA_INIT", 0.5)
        self.eval_net = DQNNetwork(
            state_size=self.num_states,
            action_size=self.num_actions,
            hidden_layers=hyperparameters.get("HIDDEN_LAYERS", [64, 32]),
            activation_fn=hyperparameters.get("ACTIVATION_FN", torch.nn.ReLU()),
            init_fn=hyperparameters.get("INIT_FN", None),
            sigma_init=self.sigma_init,
            dueling=True,
        )
        self.target_net = DQNNetwork(
            state_size=self.num_states,
            action_size=self.num_actions,
            hidden_layers=hyperparameters.get("HIDDEN_LAYERS", [64, 32]),
            activation_fn=hyperparameters.get("ACTIVATION_FN", torch.nn.ReLU()),
            init_fn=hyperparameters.get("INIT_FN", None),
            sigma_init=self.sigma_init,
            dueling=True,
        )

        self.learn_step_counter = 0
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)

        # Noise reduction and weight adjustment
        self.cumulative_reward = 0.0
        self.k_final = hyperparameters.get("K_FINAL", 1.0)
        self.inf_reward = hyperparameters.get("INF_REWARD", -200)
        self.sup_reward = hyperparameters.get("SUP_REWARD", 200)
        self.k = 0.0
        self.last_loss = None

    def act(self, state, sub_goal=None):
        """
        HRL-compatible action selection using NoisyNet and epsilon-greedy exploration.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        self.eval_net.reset_noise()
        with torch.no_grad():
            action_values = self.eval_net(state_tensor)
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            return torch.argmax(action_values).item()

    def step(self, state, action, reward, next_state, done):
        """
        Store transition and trigger learning.
        """
        self.memory.push(state, action, reward, next_state, done)
        self.cumulative_reward += reward
        if done:
            self.cumulative_reward = 0.0
        loss = None
        if len(self.memory) > self.batch_size:
            experiences, indices, weights = self.memory.sample()
            loss = self.learn(experiences, indices, weights)
            self.last_loss = loss  # Store last loss
            if self.writer:
                self.writer.add_scalar("Loss/CartPole", loss, self.learn_step_counter)
        else:
            self.last_loss = None
        return loss

    def learn(self, experiences, indices, weights):
        """
        Learn from transitions using Double DQN with PER and NROWAN enhancements.
        """
        if self.learn_step_counter % self.target_update_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        states, actions, rewards, next_states, dones = experiences
        q_eval = self.eval_net(states).gather(1, actions)
        next_actions = self.eval_net(next_states).detach().max(1)[1].unsqueeze(1)
        q_next = self.target_net(next_states).gather(1, next_actions).detach()
        q_target = rewards + self.gamma * q_next * (1 - dones)
        td_errors = q_target - q_eval

        weights_tensor = weights.to(states.device)
        loss = (weights_tensor * F.smooth_l1_loss(q_eval, q_target, reduction="none")).mean()
        abs_td_errors = torch.abs(td_errors).detach().cpu().numpy()
        self.memory.update_priorities(indices, abs_td_errors)

        D = self.eval_net.compute_noise_level_D()
        self.k = max(0.0, min(self.k_final, self.k_final * (self.cumulative_reward - self.inf_reward) / (self.sup_reward - self.inf_reward)))
        total_loss = loss + self.k * D

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.eval_net.reset_noise()
        self.target_net.reset_noise()
        # self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return total_loss.item()

    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
