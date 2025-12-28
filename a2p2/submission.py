import numpy as np
from agents import BaseAgent

class MyAgent(BaseAgent):
    """
    MyAgent: Student implementation.
    
    Implement your agent here.
    """

    def __init__(self, n_products, n_users):
        """
        Initialize your agent.

        Args:
            n_products (int): The number of products.
            n_users(int): The number of users.
        """
        super().__init__(n_products, n_users)
        self.Q = np.zeros((n_users, n_products))
        self.N = np.zeros((n_users, n_products))
        self.window = 7 * n_products
        self.epsilon = 0.12
        self.recent_rewards = [[] for _ in range(n_products)]
        self.is_fixed_env = False
        self.check_steps = 100 * n_products
        self.total_steps = 0

        # raise NotImplementedError(
        #     "MyAgent must be initialized by the student.")

    def select_action(self, obs):
        """
        Select an action given the current observation.

        Args:
            obs: observation from the environment

        Returns:
            int: chosen action index
        """
        
        # TODO: Implement your chosen action selection logic here

        # user = obs
        # if np.random.rand() < self.epsilon:
        #     action = np.random.randint(self.n_actions)
        # else:
        #     action = int(np.argmax(self.Q[user]))

        # return action
    
        user = obs
        if np.any(self.N[user] == 0):
            action = np.argmin(self.N[user])
        else:
            if np.random.rand() < self.epsilon:
                action = np.random.randint(self.n_actions)
            else:
                action = int(np.argmax(self.Q[user]))

        return action
        # raise NotImplementedError(
        #     "select_action() must be implemented by the student.")

    def update(self, obs, action, reward, next_obs):
        """
        Update your agent after receiving feedback.

        Args:
            obs: observation
            action: action taken (int)
            reward: reward received (float)
            next_obs: next observation
        """
        # TODO: Implement your chosen update logic here
        
        user = obs
        self.N[user, action] += 1
        self.total_steps += 1
        
        rewards = self.recent_rewards[action]
        rewards.append(reward)
        if len(rewards) > self.window:
            rewards.pop(0)
        recent_avg = np.mean(rewards)
        delta = abs(recent_avg - np.mean([self.Q[u, action] for u in range(self.n_users)]))

        if self.total_steps == self.check_steps:
            all_deltas = [
                abs(np.mean(self.recent_rewards[a]) - np.mean([self.Q[u, a] for u in range(self.n_users)]))
                for a in range(self.n_actions) if len(self.recent_rewards[a]) > 0
            ]

            avg_delta = np.mean(all_deltas)
            if avg_delta < 0.15: 
                self.is_fixed_env = True

        if self.is_fixed_env:
            alpha = alpha = 1/self.N[user, action]
            self.epsilon = 0.05
        if delta > 0.35:
            alpha = 2 * (1/self.N[user, action])
        else:
            alpha = 1/self.N[user, action]

        self.Q[user, action] += alpha * (reward - self.Q[user, action])
        self.epsilon = max(0.05, self.epsilon * 0.97)

        # raise NotImplementedError(
        #     "update() must be implemented by the student.")
