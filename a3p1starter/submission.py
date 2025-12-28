import numpy as np
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

def policy_evaluation_v(
    policy, transition_fn, reward_fn, gamma=0.99, theta=1e-6):
    """Implements the policy evaluation algorithm to compute V values.
    
    Args:
        policy: A numpy ndarray of shape (num_states, num_actions) denoting
            the probability of selecting an action (a) in a given state (s).
        transition_fn: A numpy ndarray of shape
            (num_states, num_actions, num_states)
            denoting the transition probabilities, T(s, a, s')
        reward_fn: A numpy ndarray of shape
            (num_states, num_actions, num_states)
            denoting the reward of R(s, a, s').
        gamma: Discount factor.
        theta: Threshold for terminating policy evaluation.
    
    Returns:
        v_value: A numpy array of shape (num_states) denoting the value
            of state (s) for the given policy.
    """
    # Ensure input dimensions are consistent.
    num_states, num_actions, _ = transition_fn.shape
    assert_equal(num_states, transition_fn.shape[2])
    assert_equal(reward_fn.shape, transition_fn.shape)
    assert_equal(policy.shape, (num_states, num_actions))
    
    # Initialize value
    v_value = np.zeros(shape=(num_states, ))
    
    # Evaluate value
    while True:
        delta = 0
        ######## PUT YOUR CODE HERE ########
        for s in range(num_states):
            v_old = v_value[s]
            v_new = 0
            for a in range(num_actions):
                action_prob = policy[s, a]
                sum_over_s_next = 0
                for s_next in range(num_states):
                    trans_prob = transition_fn[s, a, s_next]
                    reward = reward_fn[s, a, s_next]
                    sum_over_s_next += trans_prob * (reward + gamma * v_value[s_next])

                v_new += action_prob * sum_over_s_next

            v_value[s] = v_new
            delta = max(delta, abs(v_old - v_value[s]))
            
        ######## PUT YOUR CODE HERE ########
        if delta < theta:
            break
    return v_value


def policy_evaluation_q(
    policy, transition_fn, reward_fn, gamma=0.99, theta=1e-6):
    """Implements the policy evaluation algorithm to compute Q values.
    
    Args:
        policy: A numpy ndarray of shape (num_states, num_actions) denoting
            the probability of selecting an action (a) in a given state (s).
        transition_fn: A numpy ndarray of shape
            (num_states, num_actions, num_states)
            denoting the transition probabilities, T(s, a, s')
        reward_fn: A numpy ndarray of shape
            (num_states, num_actions, num_states)
            denoting the reward of R(s, a, s').
        gamma: Discount factor.
        theta: Threshold for terminating policy evaluation.
    
    Returns:
        q_value: A numpy ndarray of shape (num_states, num_actions) denoting the value
            of state (s) for the given policy.
    """
    # Ensure input dimensions are consistent.
    num_states, num_actions, _ = transition_fn.shape
    assert_equal(num_states, transition_fn.shape[2])
    assert_equal(reward_fn.shape, transition_fn.shape)
    assert_equal(policy.shape, (num_states, num_actions))
    
    # Initialize value
    q_value = np.zeros(shape=(num_states, num_actions))

    # Evaluate value
    ######## PUT YOUR CODE HERE ########
    while True:
        delta = 0
        for s in range(num_states):
            for a in range(num_actions):
                q_old = q_value[s, a]
                sum_over_s_next = 0
                for s_next in range(num_states):
                    trans_prob = transition_fn[s, a, s_next]
                    reward = reward_fn[s, a, s_next]
                    sum_over_a_next = 0
                    for a_next in range(num_actions):
                        action_prob = policy[s_next, a_next]
                        sum_over_a_next += action_prob * q_value[s_next, a_next]

                    sum_over_s_next += trans_prob * (reward + gamma * sum_over_a_next)

                q_value[s, a] = sum_over_s_next
                delta = max(delta, abs(q_old - q_value[s, a]))

        if delta < theta:
            break
    ######## PUT YOUR CODE HERE ########
    return q_value


def value_iteration(transition_fn, reward_fn, gamma=0.99, theta=1e-6):
    """Implements the value iteration algorithm.
    
    Args:
        transition_fn: A numpy ndarray of shape
            (num_states, num_actions, num_states)
            denoting the transition probabilities, T(s, a, s')
        reward_fn: A numpy ndarray of shape
            (num_states, num_actions, num_states)
            denoting the reward of R(s, a, s').
        gamma: Discount factor.
        theta: Threshold for terminating policy evaluation.
    
    Returns:
        optimal_policy: A numpy ndarray of shape
            (num_states, num_actions) denoting
            the probability of selecting an action (a) in a given state (s).
        optimal_v_value: A numpy ndarray of shape
            (num_states, ) denoting
            the optimal value of each state (s).
    """
    # Ensure input dimensions are consistent.
    num_states, num_actions, _ = transition_fn.shape
    assert_equal(num_states, transition_fn.shape[2])
    assert_equal(reward_fn.shape, transition_fn.shape)
    
    # Initialize value
    v_value = np.zeros(shape=(num_states, ))
    
    # Perform value iteration to compute optimal V
    num_iteration = 0
    while True:
        delta = 0
        ######## PUT YOUR CODE HERE ########
        for s in range(num_states):
            v_old = v_value[s]
            q_values_in_s = np.zeros(shape=(num_actions, ))
            for a in range(num_actions):
                sum_over_s_next = 0
                for s_next in range(num_states):
                    trans_prob = transition_fn[s, a, s_next]
                    reward = reward_fn[s, a, s_next]
                    sum_over_s_next += trans_prob * (reward + gamma * v_value[s_next])

                q_values_in_s[a] = sum_over_s_next

            v_value[s] = q_values_in_s.max()
            delta = max(delta, abs(v_old - v_value[s]))
    
        ######## PUT YOUR CODE HERE ########
        num_iteration += 1
        if delta < theta:
            break
    optimal_v_value = v_value
    
    # Compute optimal Q from optimal V
    optimal_q_value = np.zeros(shape=(num_states, num_actions)) 
    ######## PUT YOUR CODE HERE ########
    for s in range(num_states):
        for a in range(num_actions):
            sum_over_s_next = 0
            for s_next in range(num_states):
                trans_prob = transition_fn[s, a, s_next]
                reward = reward_fn[s, a, s_next]
                sum_over_s_next += trans_prob * (reward + gamma * optimal_v_value[s_next])
            
            optimal_q_value[s, a] = sum_over_s_next

    ######## PUT YOUR CODE HERE ########
   
    # Compute optimal policy from optimal Q
    optimal_policy = np.zeros(shape=(num_states, num_actions))
    for s in range(num_states):
        optimal_action_in_s = optimal_q_value[s].argmax()
        optimal_policy[s, optimal_action_in_s] = 1.
    
    return (optimal_policy, optimal_v_value)


def policy_iteration(transition_fn, reward_fn, gamma=0.99, theta=1e-6):
    """Implements the policy iteration algorithm.
    
    Args:
        transition_fn: A numpy ndarray of shape
            (num_states, num_actions, num_states)
            denoting the transition probabilities, T(s, a, s')
        reward_fn: A numpy ndarray of shape
            (num_states, num_actions, num_states)
            denoting the reward of R(s, a, s').
        gamma: Discount factor.
    
    Returns:
        optimal_policy: A numpy ndarray of shape
            (num_states, num_actions) denoting
            the probability of selecting an action (a) in a given state (s).
        optimal_v_value: A numpy ndarray of shape
            (num_states, ) denoting
            the optimal value of each state (s).
    """
    # Ensure input dimensions are consistent.
    num_states, num_actions, _ = transition_fn.shape
    assert_equal(num_states, transition_fn.shape[2])
    assert_equal(reward_fn.shape, transition_fn.shape)
    
    # Initialize policy
    policy = np.zeros(shape=(num_states, num_actions))
    
    # Perform policy iteration
    num_iteration = 0
    while True:
        delta = 0
        
        # Policy Evaluation
        ######## PUT YOUR CODE HERE ########
        q_value = policy_evaluation_q(policy, transition_fn, reward_fn, gamma, theta)
        ######## PUT YOUR CODE HERE ########
        
        # Policy Improvement
        ######## PUT YOUR CODE HERE ########
        policy_stable = True
        for s in range(num_states):
            old_action = policy[s].argmax()
            best_action = q_value[s].argmax()
            if old_action != best_action:
                policy_stable = False
            policy[s] = np.eye(num_actions)[best_action]
        
        ######## PUT YOUR CODE HERE ########
        num_iteration += 1
        # Termination Condition
        ######## PUT YOUR CODE HERE ########
        if policy_stable:
            break
        ######## PUT YOUR CODE HERE ########

    # Compute optimal V value
    optimal_policy = policy.copy()
    optimal_v_value = policy_evaluation_v(
        policy, transition_fn, reward_fn, gamma)
    return (optimal_policy, optimal_v_value)
