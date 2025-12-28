import gymnasium as gym

import numpy as np

from submission import MonteCarloAgent, QLearningAgent


if __name__ == "__main__":
    # Initialize the environment
    env = gym.make("FrozenLake-v1", is_slippery="True")
    num_actions = env.action_space.n
    env_step = env.step
    env_reset = env.reset

    # Monte Carlo agent
    mc_agent = MonteCarloAgent(num_actions, env_step, env_reset)
    mc_agent.train(100001, make_plot=True)

    # Q Learning agent
    q_agent = QLearningAgent(num_actions, env_step, env_reset)
    q_agent.train(100001, make_plot=True)
