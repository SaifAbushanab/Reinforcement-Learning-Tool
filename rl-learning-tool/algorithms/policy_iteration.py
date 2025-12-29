"""
Policy Iteration Algorithm

Finds optimal policy by alternating between:
1. Policy Evaluation: Compute V(s) for current policy
2. Policy Improvement: Make policy greedy w.r.t. V(s)
"""

import numpy as np


def policy_iteration(env, gamma=0.99, theta=0.0001, max_iterations=100):
    """
    Find optimal policy using Policy Iteration.
    
    Args:
        env: Environment with get_transition_prob() method
        gamma: Discount factor
        theta: Convergence threshold
        max_iterations: Maximum policy improvement iterations
        
    Yields:
        Dict with policy, values, iteration info
    """
    n_states = env.n_states
    n_actions = env.n_actions
    
    policy = np.ones((n_states, n_actions)) / n_actions
    V = np.zeros(n_states)
    
    for iteration in range(max_iterations):
        # Step 1: Policy Evaluation
        while True:
            delta = 0
            for state in range(n_states):
                old_value = V[state]
                new_value = 0
                for action in range(n_actions):
                    action_prob = policy[state][action]
                    for prob, next_state, reward, done in env.get_transition_prob(state, action):
                        if done:
                            new_value += action_prob * prob * reward
                        else:
                            new_value += action_prob * prob * (reward + gamma * V[next_state])
                V[state] = new_value
                delta = max(delta, abs(old_value - new_value))
            if delta < theta:
                break
        
        # Step 2: Policy Improvement
        policy_stable = True
        for state in range(n_states):
            old_action = np.argmax(policy[state])
            action_values = np.zeros(n_actions)
            for action in range(n_actions):
                for prob, next_state, reward, done in env.get_transition_prob(state, action):
                    if done:
                        action_values[action] += prob * reward
                    else:
                        action_values[action] += prob * (reward + gamma * V[next_state])
            best_action = np.argmax(action_values)
            policy[state] = np.zeros(n_actions)
            policy[state][best_action] = 1.0
            if old_action != best_action:
                policy_stable = False
        
        yield {
            "values": V.copy(),
            "policy": policy.copy(),
            "iteration": iteration + 1,
            "policy_stable": policy_stable,
        }
        
        if policy_stable:
            break
    
    return policy, V
