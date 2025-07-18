import torch
import torch.nn.functional as F
import numpy as np
from architectures.DQNNROWAN.noisy_linear import NoisyLinear
from architectures.DQNNROWAN.dqn_network import DQNNetwork
from shared.replay_buffer import ReplayMemory
from shared.abstract_policy import AbstractPolicy

class DQNAgentCartPole(AbstractPolicy):
    def __init__(self, state_size, action_size, hyperparameters, writer=None):
        """
        Initialize the DQNNROWAN agent for CartPole.
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

        # Replay memory
        self.memory = ReplayMemory(hyperparameters["MEMORY_CAPACITY"], self.batch_size)

        # Initialize networks
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

        # Noise reduction and weight adjustment
        self.cumulative_reward = 0.0
        self.k_final = hyperparameters.get('K_FINAL', 1.0)  # Final value of k
        self.inf_reward = hyperparameters.get('INF_REWARD', -200)
        self.sup_reward = hyperparameters.get('SUP_REWARD', 200)
        self.k = 0.0  # Initialize k
        self.last_loss = None

    def act(self, state, sub_goal=None):
        """
        Action selection with NoisyNet exploration and HRL compatibility.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        if self.use_noisy:
            self.eval_net.reset_noise()
        with torch.no_grad():
            action_values = self.eval_net(state_tensor)
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            return torch.argmax(action_values).item()
        # return torch.argmax(action_values).item()

    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        self.cumulative_reward += reward
        if done:
            self.cumulative_reward = 0.0

        # Learn from memory if enough samples are available
        loss = None
        if len(self.memory) > self.batch_size:
            loss = self.learn()
            self.last_loss = loss  # Store last loss
            if self.writer:
                self.writer.add_scalar('Loss/CartPole', loss, self.learn_step_counter)
        else:
            self.last_loss = None
        return loss

    def learn(self):
        if self.learn_step_counter % self.target_update_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        states, actions, rewards, next_states, dones = self.memory.sample()
        q_eval = self.eval_net(states).gather(1, actions)
        q_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        q_target = rewards + self.gamma * q_next * (1 - dones)

        td_errors = q_target - q_eval
        loss = self.loss_func(q_eval, q_target)

        D = self.eval_net.compute_noise_level_D()

        # Adjust k based on cumulative reward
        self.k = self.k_final * (self.cumulative_reward - self.inf_reward) / (self.sup_reward - self.inf_reward)
        self.k = max(0.0, min(self.k_final, self.k))

        total_loss = loss + self.k * D  # Combine TD loss with noise reduction term
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        if self.use_noisy:
            self.eval_net.reset_noise()
            self.target_net.reset_noise()
        # self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return total_loss.item()

    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
