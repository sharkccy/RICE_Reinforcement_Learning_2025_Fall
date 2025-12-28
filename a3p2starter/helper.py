import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


class BaseAgent:
    """
    Base Class for agents.
    Students will extend this class when implementing algorithms.
    """
    def __init__(self, num_actions, env_step, env_reset):
        """
        Initializes the agent.

        Args:
            num_actions: Number of actions available to the agent.
                We assume agent has access to all actions in each state.
            env_step: Step function of the environment. For more details, see
                https://gymnasium.farama.org/api/env/#gymnasium.Env.step
            env_reset: Reset function of the environment. For more details, see
                https://gymnasium.farama.org/api/env/#gymnasium.Env.reset
        """
        self.num_actions = num_actions
        self.env_step = env_step
        self.env_reset = env_reset
        self.policy = self.make_policy()
        self.behavior_policy = self.make_behavior_policy()

    def make_policy(self):
        """
        Return a policy function that will be used for evaluation. The policy
        takes observation as input and returns an action.
        """
        raise NotImplementedError

    def make_behavior_policy(self):
        """
        Similar to make_policy, it returns a policy function. But this one used
        for interaction with the environment during training.
        """
        raise NotImplementedError

    def run_episode(self, episode_policy):
        """
        Generate one episode with the given policy
        """
        episode = []
        done = False
        obs, _ = self.env_reset()
        episode_return = 0
        while not done:
            action = episode_policy(obs)
            next_obs, reward, terminated, truncated, _ = self.env_step(action)
            done = terminated or truncated
            episode.append([obs, action, reward, next_obs, done])
            obs = next_obs
            episode_return += reward

        return (episode, episode_return)

    def evaluate(self, num_eval_episodes=1000):
        """Evaluates the agent."""
        list_returns = []
        average_return = 0
        for episode_idx in range(num_eval_episodes):
            _, episode_return = self.run_episode(self.policy)
            average_return += (
                (1. / (episode_idx+1)) * (episode_return - average_return))
            list_returns.append(episode_return)

        average_return = round(np.mean(list_returns), 3)
        stdev_return = round(np.std(list_returns), 3)
        return average_return, stdev_return
