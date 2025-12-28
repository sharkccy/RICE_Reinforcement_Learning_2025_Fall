import numpy as np

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    Abstract Base Class for agents.
    Students will extend this class when implementing algorithms.
    """

    def __init__(self, n_products, n_users):
        """
        Initializes the agent.
        Args:
            n_products (int): The number of products.
            n_users(int): The number of users.
        """
        self.n_actions = n_products
        self.n_users = n_users

    @abstractmethod
    def select_action(self, obs):
        """
        Select an action given the current observation.
        Args:
            obs: observation from environment
        Returns:
            int: chosen action index
        """
        pass

    @abstractmethod
    def update(self, obs, action, reward, next_obs):
        """
        Update the agent based on the feedback.
        Args:
            obs: observation
            action: action taken
            reward: reward received
            next_obs: next observation
        """
        pass


class RandomAgent(BaseAgent):
    """
    Random Agent:
    - Selects an action uniformly at random.
    - Ignores observations and rewards.

    This agent is provided as a baseline for comparison.
    """

    def __init__(self, n_products, n_users):
        super().__init__(n_products, n_users)

    def select_action(self, obs):
        return np.random.randint(self.n_actions)

    def update(self, obs, action, reward, next_obs):
        # Random agent does not learn from feedback
        pass
