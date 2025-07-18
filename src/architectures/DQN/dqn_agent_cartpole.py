from shared.abstract_policy import AbstractPolicy
from architectures.DQN.dqn_network import DQNNetwork
from shared.replay_buffer import ReplayMemory
import torch
import numpy as np
import torch.nn.functional as F

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

        # Initialize networks
        self.eval_net = DQNNetwork(
            state_size=self.num_states,
            action_size=self.num_actions,
            hidden_layers=hyperparameters.get("HIDDEN_LAYERS", [64, 32]),
            activation_fn=hyperparameters.get("ACTIVATION_FN", F.relu),
            init_fn=hyperparameters.get("INIT_FN", None),
        )
        self.target_net = DQNNetwork(
            state_size=self.num_states,
            action_size=self.num_actions,
            hidden_layers=hyperparameters.get("HIDDEN_LAYERS", [64, 32]),
            activation_fn=hyperparameters.get("ACTIVATION_FN", F.relu),
            init_fn=hyperparameters.get("INIT_FN", None),
        )

        self.learn_step_counter = 0
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)
        self.loss_func = F.smooth_l1_loss
        self.last_loss = None

    def act(self, state, sub_goal=None):
        """
        Select an action using epsilon-greedy policy.
        Optionally consider sub_goal for HRL compatibility.
        """
        if np.random.rand() < self.epsilon:
            # Random exploration
            action = np.random.randint(0, self.num_actions)
        else:
            # Exploitation using the evaluation network
            state = torch.unsqueeze(torch.FloatTensor(state), 0)
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]
        return action

    def step(self, state, action, reward, next_state, done):
        """
        Integrates storing transitions and learning.
        Returns the loss value for TensorBoard logging.
        """
        # Store the transition
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
        Learn from the sampled transitions.
        Returns the loss value for TensorBoard logging.
        """
        # Update the target network periodically
        if self.learn_step_counter % self.target_update_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # Sample a mini-batch of transitions
        states, actions, rewards, next_states, dones = self.memory.sample()

        # Compute Q-values
        q_eval = self.eval_net(states).gather(1, actions)
        q_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        q_target = rewards + self.gamma * q_next * (1 - dones)

        # Compute loss
        loss = self.loss_func(q_eval, q_target)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return loss.item()

    def update_epsilon(self):
        """
        Decay epsilon after each episode.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
