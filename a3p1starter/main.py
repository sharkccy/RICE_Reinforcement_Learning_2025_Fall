import gymnasium as gym

import numpy as np

from helper import get_mdp_parameters, render_policy
from submission import value_iteration, policy_iteration


if __name__ == "__main__":
    # Initialize the environment
    env = gym.make("FrozenLake-v1",
        is_slippery=True,
        render_mode="human")

    # Retrieve MDP parameters
    num_states, num_actions, transition_fn, reward_fn = get_mdp_parameters(env)

    # Computer optimal policy and value using value iteration
    vi_optimal_policy, vi_optimal_v_value = value_iteration(
        transition_fn, reward_fn, gamma=0.99)

    # Render optimal policy computed using value iteration
    try:
        render_policy(env, vi_optimal_policy)
    except Exception as e:
        print(e)
        print("Either your python runtime does not support rendering...")
        print("or your pygame installation is incorrect.")

    # Computer optimal policy and value using policy iteration
    pi_optimal_policy, pi_optimal_v_value = policy_iteration(
        transition_fn, reward_fn, gamma=0.99)

    # Render optimal policy computed using policy iteration
    try:
        render_policy(env, pi_optimal_policy)
    except Exception as e:
        print(e)
        print("Either your python runtime does not support rendering...")
        print("or your pygame installation is incorrect.")
