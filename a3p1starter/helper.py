import gymnasium as gym

import numpy as np
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal


def get_mdp_parameters(env):
    """Helper method to retrieve MDP parameters.
    
    Args:
        env: Instance of the Frozen Lake MDP environment.

    Returns:
        num_states: Int, number of states.
        num_actions: Int, number of actions.
        transition_fn: A numpy ndarray of shape
            (num_states, num_actions, num_states)
            denoting the transition probabilities, T(s, a, s')
        reward_fn: A numpy ndarray of shape
            (num_states, num_actions, num_states)
            denoting the reward of R(s, a, s').
    """
    # Number of states and actions
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Helper variable: mapping from state to grid type
    unwrapped_env = env.unwrapped
    flattened_map = unwrapped_env.desc.flatten()

    # Initialize Transition and Reward functions
    transition_fn = np.zeros(
        shape=(num_states, num_actions, num_states))
    reward_fn = np.zeros(
        shape=(num_states, num_actions, num_states))

    # Fill Transition and Reward functions
    for s in range(num_states):
        terminal = flattened_map[s] in [b"H", b"G"]
        for a in range(num_actions):
            if s is terminal:
                transition_fn[s, a, s] = 1.
                continue
            for prob_next_s, next_s, r, _ in unwrapped_env.P[s][a]:
                transition_fn[s, a, next_s] += prob_next_s
                reward_fn[s, a, next_s] = r


    # Ensure Transition function is indeed a probability distribution
    for s in range(num_states):
        for a in range(num_actions):
            assert_almost_equal(transition_fn[s,a].sum(), 1., decimal=7)
    
    # Return as a tuple
    return (num_states, num_actions, transition_fn, reward_fn)


def render_policy(env, policy):
    """Renders a policy for the Frozen Lake environment.
    
    Args:
        env: Instance of the Frozen Lake MDP environment.
        policy: A numpy ndarray of shape (num_states, num_actions) denoting
            the probability of selecting an action (a) in a given state (s).
    
    Returns:
        None
    """
    assert_equal(policy.shape, (env.observation_space.n, env.action_space.n))
    state, _ = env.reset()
    while True:
        action = policy[state].argmax()
        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if done:
            env.close()
            break
        state = next_state

    return
