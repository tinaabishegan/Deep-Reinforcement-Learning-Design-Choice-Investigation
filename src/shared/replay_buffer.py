# replay_buffer.py

import random
import numpy as np
import torch
from collections import namedtuple, deque

class ReplayMemory:
    def __init__(self, capacity, batch_size, seed=42):
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def push(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])
        ).float()
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        ).long()
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])
        ).float()
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])
        ).float()
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        ).float()

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class PrioritizedReplayMemory:
    def __init__(self, capacity, batch_size, seed=42, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.batch_size = batch_size
        self.alpha = alpha  # How much prioritization is used (0 - no prioritization, 1 - full prioritization)
        self.beta_start = beta_start  # Initial value of beta for importance sampling
        self.beta_frames = beta_frames  # Number of frames over which beta will be annealed from beta_start to 1.0
        self.beta = beta_start
        self.frame = 1

        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done"])
        self.memory = []
        self.pos = 0

        # For priorities
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        random.seed(seed)

    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.memory else 1.0

        e = self.experience(state, action, reward, next_state, done)

        if len(self.memory) < self.capacity:
            self.memory.append(e)
        else:
            self.memory[self.pos] = e

        self.priorities[self.pos] = max_priority  # Assign max priority to new experience
        self.pos = (self.pos + 1) % self.capacity

    def sample(self):
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]

        # Calculate probabilities
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Sample indices based on probabilities
        indices = np.random.choice(len(self.memory), self.batch_size, p=probabilities)
        experiences = [self.memory[idx] for idx in indices]

        # Importance Sampling weights
        total = len(self.memory)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize for stability
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)

        # Anneal beta towards 1.0
        self.beta = min(1.0, self.beta + (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1

        # Extract tensors
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float()

        # Pack experiences into a tuple
        experiences = (states, actions, rewards, next_states, dones)

        return (experiences, indices, weights)

    def update_priorities(self, indices, td_errors):
        # Ensure td_errors is a NumPy array
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().cpu().numpy()
        
        # Remove unnecessary dimensions
        td_errors = td_errors.squeeze()

        # Convert td_errors to priorities
        priorities = np.abs(td_errors) + 1e-6  # Add epsilon to avoid zero priority
        self.priorities[indices] = priorities


    def __len__(self):
        return len(self.memory)