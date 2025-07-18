# shared/abstract_policy.py

from abc import ABC, abstractmethod

class AbstractPolicy(ABC):
    @abstractmethod
    def act(self, state, sub_goal=None):
        """
        Select an action for the given state and sub-goal.
        """
        pass

    @abstractmethod
    def step(self, state, action, reward, next_state, done):
        """
        Update the policy with a transition.
        """
        pass

    def update_epsilon(self):
        """
        Update epsilon (if applicable).
        """
        pass  # Not all agents use epsilon

    def get_epsilon(self):
        """
        Get current epsilon value (if applicable).
        """
        return None  # Not all agents use epsilon
