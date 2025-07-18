import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from architectures.DoubleDQNPER.dqn_network import DQNNetwork
from shared.replay_buffer import PrioritizedReplayMemory
from shared.abstract_policy import AbstractPolicy


class DQNAgentLunarLander(AbstractPolicy):
    def __init__(self, state_size, action_size, hyperparameters, seed=42, writer=None):
        """
        Initialize the Double DQN agent for LunarLander with PER and HRL compatibility.
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

        # Prioritized Replay Memory
        self.memory = PrioritizedReplayMemory(
            hyperparameters["MEMORY_CAPACITY"],
            self.batch_size,
            seed=42,
            alpha=hyperparameters.get("PER_ALPHA", 0.6),
            beta_start=hyperparameters.get("PER_BETA_START", 0.4),
            beta_frames=hyperparameters.get("PER_BETA_FRAMES", 100000),
        )

        # Q-Networks
        self.qnetwork_local = DQNNetwork(
            state_size,
            action_size,
            hidden_layers=hyperparameters.get("HIDDEN_LAYERS", [64, 64]),
            activation_fn=hyperparameters.get("ACTIVATION_FN", F.relu),
            init_fn=hyperparameters.get("INIT_FN", None),
            seed=seed,
        )
        self.qnetwork_target = DQNNetwork(
            state_size,
            action_size,
            hidden_layers=hyperparameters.get("HIDDEN_LAYERS", [64, 64]),
            activation_fn=hyperparameters.get("ACTIVATION_FN", F.relu),
            init_fn=hyperparameters.get("INIT_FN", None),
            seed=seed,
        )
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=hyperparameters['LEARNING_RATE'])
        self.t_step = 0
        self.last_loss = None

    def act(self, state, sub_goal=None):
        """
        HRL-compatible action selection using epsilon-greedy strategy.
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
        Store transition and trigger learning if ready.
        """
        loss = None
        self.memory.push(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.target_update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences, indices, weights = self.memory.sample()
            loss = self.learn(experiences, indices, weights)
            self.last_loss = loss  # Store last loss 
            if self.writer:
                self.writer.add_scalar('Loss/LunarLander', loss, self.t_step)
        else:
            self.last_loss = None
        return loss

    def learn(self, experiences, indices, weights):
        """
        Learn from the sampled transitions using Double DQN and PER.
        """
        states, actions, rewards, next_states, dones = experiences

        # Double DQN logic
        q_eval = self.qnetwork_local(states).gather(1, actions)
        next_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        q_next = self.qnetwork_target(next_states).gather(1, next_actions).detach()
        q_target = rewards + (self.gamma * q_next * (1 - dones))

        # Compute TD errors and weighted loss
        td_errors = q_target - q_eval
        weights_tensor = torch.tensor(weights, dtype=torch.float32).to(states.device)
        loss = (weights_tensor * F.smooth_l1_loss(q_eval, q_target, reduction='none')).mean()

        # Update priorities in replay buffer
        abs_td_errors = torch.abs(td_errors).detach().cpu().numpy()
        self.memory.update_priorities(indices, abs_td_errors)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update of target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

        return loss.item()

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)
