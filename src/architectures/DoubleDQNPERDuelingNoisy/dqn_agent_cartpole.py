import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from architectures.DoubleDQNPERDuelingNoisy.noisy_linear import NoisyLinear
from architectures.DoubleDQNPERDuelingNoisy.dqn_network import DQNNetwork
from shared.replay_buffer import PrioritizedReplayMemory
from shared.abstract_policy import AbstractPolicy

class DQNAgentCartPole(AbstractPolicy):
    def __init__(self, state_size, action_size, hyperparameters, writer=None):
        """
        HRL-enhanced Double DQN with PER, Dueling Network, and Noisy Layers for CartPole.
        """
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

        # Noisy network parameters
        self.sigma_init = hyperparameters.get("SIGMA_INIT", 0.5)

        # Initialize networks
        self.eval_net = DQNNetwork(
            state_size=self.num_states,
            action_size=self.num_actions,
            hidden_layers=hyperparameters.get("HIDDEN_LAYERS", [64, 32]),
            activation_fn=hyperparameters.get("ACTIVATION_FN", nn.ReLU()),
            init_fn=hyperparameters.get("INIT_FN", None),
            sigma_init=self.sigma_init,
            dueling=True,
        )
        self.target_net = DQNNetwork(
            state_size=self.num_states,
            action_size=self.num_actions,
            hidden_layers=hyperparameters.get("HIDDEN_LAYERS", [64, 32]),
            activation_fn=hyperparameters.get("ACTIVATION_FN", nn.ReLU()),
            init_fn=hyperparameters.get("INIT_FN", None),
            sigma_init=self.sigma_init,
            dueling=True,
        )

        self.learn_step_counter = 0
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)
        self.last_loss = None

    def act(self, state, sub_goal=None):
        """
        HRL-compatible action selection using Noisy Layers and epsilon-greedy strategy.
        """
        self.eval_net.reset_noise()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.eval_net(state_tensor)
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            return torch.argmax(action_values).item()

    def step(self, state, action, reward, next_state, done):
        """
        Store transition and learn if ready.
        """
        self.memory.push(state, action, reward, next_state, done)
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
        Double DQN learning with PER and Noisy Layers.
        """
        if self.learn_step_counter % self.target_update_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        states, actions, rewards, next_states, dones = experiences

        # Compute Q-values
        q_eval = self.eval_net(states).gather(1, actions)
        next_actions = self.eval_net(next_states).detach().max(1)[1].unsqueeze(1)
        q_next = self.target_net(next_states).gather(1, next_actions).detach()
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

        self.adjust_noise_scale(td_errors)
        self.eval_net.reset_noise()
        self.target_net.reset_noise()

        # self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return loss.item()

    def adjust_noise_scale(self, td_errors):
        """
        Adjust noise scale in Noisy Layers based on TD error.
        """
        mean_td_error = td_errors.abs().mean().item()
        new_scale = max(0.01, min(1.0, mean_td_error))
        if new_scale == 0.1:
            print("1 noise scale now")
            if new_scale == 0.01:
                print("min noise scale now")
        for module in self.eval_net.modules():
            if isinstance(module, NoisyLinear):
                module.set_noise_scale(new_scale)
        for module in self.target_net.modules():
            if isinstance(module, NoisyLinear):
                module.set_noise_scale(new_scale)

    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
