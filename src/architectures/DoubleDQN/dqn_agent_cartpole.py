import torch
import torch.nn.functional as F
import numpy as np
from architectures.DoubleDQN.dqn_network import DQNNetwork
from shared.replay_buffer import ReplayMemory
from shared.abstract_policy import AbstractPolicy

class DQNAgentCartPole(AbstractPolicy):
    def __init__(self, state_size, action_size, hyperparameters, writer=None):
        """
        Initialize the Double DQN agent for CartPole with HRL compatibility.
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
        HRL-compatible action selection using epsilon-greedy strategy.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_values = self.eval_net(state_tensor)
            return torch.argmax(action_values).item()

    def step(self, state, action, reward, next_state, done):
        """
        Store transition and trigger learning if ready.
        """
        self.memory.push(state, action, reward, next_state, done)
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
        """
        Perform a learning step using Double DQN logic.
        """
        if self.learn_step_counter % self.target_update_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        states, actions, rewards, next_states, dones = self.memory.sample()

        # Double DQN logic
        q_eval = self.eval_net(states).gather(1, actions)
        next_actions = self.eval_net(next_states).detach().max(1)[1].unsqueeze(1)
        q_next = self.target_net(next_states).gather(1, next_actions).detach()
        q_target = rewards + self.gamma * q_next * (1 - dones)

        loss = self.loss_func(q_eval, q_target)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return loss.item()

    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
