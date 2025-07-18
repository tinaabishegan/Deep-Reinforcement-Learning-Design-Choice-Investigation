import torch
import torch.nn.functional as F
import numpy as np
from architectures.DQNNoisy.noisy_linear import NoisyLinear
from architectures.DQNNoisy.dqn_network import DQNNetwork
from shared.abstract_policy import AbstractPolicy
from shared.replay_buffer import ReplayMemory


class DQNAgentCartPole(AbstractPolicy):
    def __init__(self, state_size, action_size, hyperparameters):
        """
        Initialize the DQN Agent for CartPole.
        """
        self.num_states = state_size
        self.num_actions = action_size

        # Hyperparameters
        self.batch_size = hyperparameters["BATCH_SIZE"]
        self.learning_rate = hyperparameters["LEARNING_RATE"]
        self.gamma = hyperparameters["GAMMA"]
        self.epsilon = hyperparameters["EPSILON_START"]
        self.min_epsilon = hyperparameters["EPSILON_END"]
        self.epsilon_decay = hyperparameters["EPSILON_DECAY"]
        self.target_update_iter = hyperparameters["TARGET_UPDATE_EVERY"]

        # Replay memory
        self.memory = ReplayMemory(hyperparameters["MEMORY_CAPACITY"], self.batch_size)

        # Initialize networks with noisy layers if enabled
        self.use_noisy = hyperparameters.get("USE_NOISY", False)
        self.noise_scale = hyperparameters.get("NOISE_SCALE", 0.5)

        self.eval_net = DQNNetwork(
            state_size=self.num_states,
            action_size=self.num_actions,
            hidden_layers=hyperparameters.get("HIDDEN_LAYERS", [64, 32]),
            activation_fn=hyperparameters.get("ACTIVATION_FN", F.relu),
            init_fn=hyperparameters.get("INIT_FN", None),
            use_noisy=self.use_noisy,
            sigma_init=self.noise_scale,
        )
        self.target_net = DQNNetwork(
            state_size=self.num_states,
            action_size=self.num_actions,
            hidden_layers=hyperparameters.get("HIDDEN_LAYERS", [64, 32]),
            activation_fn=hyperparameters.get("ACTIVATION_FN", F.relu),
            init_fn=hyperparameters.get("INIT_FN", None),
            use_noisy=self.use_noisy,
            sigma_init=self.noise_scale,
        )

        self.learn_step_counter = 0
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)
        self.loss_func = F.smooth_l1_loss
        self.last_loss = None

    def act(self, state, sub_goal=None):
        """
        Hybrid action selection using epsilon-greedy and noisy networks.
        Optionally consider sub_goal for HRL compatibility.
        """
        if self.use_noisy:
            self.eval_net.reset_noise()
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_values = self.eval_net(state_tensor)

            if np.random.rand() < self.epsilon:
                return np.random.randint(0, self.num_actions)
            else:
                return torch.argmax(action_values).item()
        else:
            if np.random.rand() < self.epsilon:
                return np.random.randint(0, self.num_actions)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action_values = self.eval_net(state_tensor)
                return torch.argmax(action_values).item()

    def step(self, state, action, reward, next_state, done):
        """
        Save transition to memory and learn periodically.
        Returns the loss value for TensorBoard logging.
        """
        self.memory.push(state, action, reward, next_state, done)
        # Learn from memory if enough samples are available
        loss = None
        if len(self.memory) > self.batch_size:
            loss = self.learn()
            self.last_loss = loss  # Store last loss
        else:
            self.last_loss = None
        return loss

    def learn(self):
        """
        Learn from replay memory.
        Returns the loss value for metrics tracking.
        """
        if self.learn_step_counter % self.target_update_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        states, actions, rewards, next_states, dones = self.memory.sample()

        q_eval = self.eval_net(states).gather(1, actions)
        q_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        q_target = rewards + self.gamma * q_next * (1 - dones)

        td_errors = q_target - q_eval
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.use_noisy:
            self.adjust_noise_scale(td_errors)
            self.eval_net.reset_noise()
            self.target_net.reset_noise()
        # self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return loss.item()

    def adjust_noise_scale(self, td_errors):
        mean_td_error = td_errors.abs().mean().item()
        new_scale = max(0.1, min(1.0, mean_td_error))
        for layer in self.eval_net.layers:
            if isinstance(layer, NoisyLinear):
                layer.set_noise_scale(new_scale)
        for layer in self.target_net.layers:
            if isinstance(layer, NoisyLinear):
                layer.set_noise_scale(new_scale)

    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
