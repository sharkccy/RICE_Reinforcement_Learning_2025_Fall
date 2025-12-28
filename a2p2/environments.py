import gymnasium as gym
import numpy as np
from gymnasium import spaces


class BaseEnv(gym.Env):
    """
    Base class for simplified product recommendation scenarios, based on
    Gymnasium's (an API standard for reinforcement learning) Env class.
    For more information on Gymnasium and its Env class, please see:
    https://gymnasium.farama.org/api/env/

    At each step,
        - the agent must select one product
        - this product is then recommended to the user
        - the user then decides whether to "click" the product or not.

    The step function of the enviroment simulates this interaction.
    The step function
        - takes agent's action as input,
        - provides a reward based on whether the user "clicks" the product, and
        - provides an observation about the next interaction.

    The agent's goal is to maximize its cumulative reward.
    """
    def __init__(self, n_products=10, n_users=3):
        """
        Initializes the environment.

        Args:
            n_products (int): The number of products.
            n_users(int): The number of users.
        """
        super().__init__()
        self.n_products = n_products
        self.n_users = n_users
        self.action_space = spaces.Discrete(n_products)
        self.observation_space = spaces.Discrete(n_users)
        self.user = np.random.randint(self.n_users)

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.
        """
        super().reset(seed=seed)
        self.user = 0
        return 0, {}

    def step(self, action: int):
        """
        Executes one time step in the environment.

        The agent chooses an action (a product).
        Then the environment returns
            - reward: a reward based on whether the user "clicks" the product,
            - obs: an observation about the next interaction, and
            - other variables that are required by the gym.Env class.
        For this assignment, your agent
            - needs to consider only reward and obs
            - it can ignore the other variables returned by the step function.

        Args:
            action (int): The index of the product chosen by the agent.

        Returns:
            Tuple[int, int, bool, bool, Dict[str, Any]]: A tuple containing
                - the observation,
                - the reward (1 for a click, 0 otherwise),
                - a Boolean indicating if the episode is done,
                - a Boolean for truncation, and
                - an empty info dictionary.
        """
        assert self.action_space.contains(action)

        # Simulate click and assign reward
        click = self.simulate_user_click(action)
        if click:
            reward = 1.0
        else:
            reward = 0.0

        # Select next user and assign observation
        self.select_next_user()
        obs = self.user

        # Return obs, reward, and other vars required by gym.Env
        return obs, reward, False, False, {}

    def simulate_user_click(self, action: int):
        """
        Simulates whether the user clicks the recommended product.

        Args:
            action (int): The index of the product chosen by the agent.

        Returns:
            a Boolean indicating if the user clicks the product.
        """
        raise NotImplementedError()

    def select_next_user(self):
        """
        Selects the next user.
        """
        raise NotImplementedError()

    def render(self):
        """
        Renders the environment.
        """
        pass


class ValidationEnv(BaseEnv):
    """
    A simple environment for validating your agent.

    In this environment,
        - the click probabilities are conditioned only on the product
        - these conditional probabilities of clicks do not change with time
    """

    def __init__(self, n_products=10, n_users=3):
        """
        Initializes the ValidationEnv.

        Args:
            n_products (int): The number of products.
            n_users(int): The number of users.
        """
        super().__init__(n_products, n_users)
        # True click probabilities (hidden from the agent)
        self.probs = np.random.rand(n_products)

    def simulate_user_click(self, action: int):
        """
        Simulates whether the user clicks the recommended product,
        based on the true click probabilities.

        Args:
            action (int): The index of the product chosen by the agent.

        Returns:
            a Boolean indicating if the user clicks the product.
        """
        click = np.random.binomial(1, self.probs[action])
        return click

    def select_next_user(self):
        """
        Selects the next user,
        based on uniform random distribution.
        """
        self.user = np.random.randint(self.n_users)

    def render(self):
        print(f"True click probabilities: {self.probs}")
