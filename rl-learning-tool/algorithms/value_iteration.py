"""
Value Iteration Algorithm

Computes optimal value function using Bellman Optimality Equation.

Update Rule:
    V(s) = max_a Σ_s' P(s'|s,a) * [R + γ*V(s')]
Extract Policy:
    π(s) = argmax_a Σ_s' P(s'|s,a) * [R + γ*V(s')]
"""

import numpy as np
def value_iteration(env, gamma=0.99, theta=0.0001, max_iterations=100):
    """
    Find optimal value function using Value Iteration.
    
    Args:
        env: Environment
        gamma: Discount factor
        theta: Convergence threshold
        max_iterations: Maximum iterations
        
    Yields:
        Dict with values, policy, iteration info
    """

    V = np.zeros(env.n_states)
    
    for i in range(max_iterations):
        delta = 0
        
        # Update each state's value
        for s in range(env.n_states):
            v = V[s]
            q_values = np.zeros(env.n_actions) # Initialize q_values for state s
            
            # Calculate value for each action in the selected state
            for a in range(env.n_actions):
                for prob, next_s, reward, done in env.get_transition_prob(s, a):
                    # Bellman equation: R + gamma * V(s')
                    if done:
                        # If the state is terminal, there is no future value
                        future_value = 0
                    else:
                        future_value = gamma * V[next_s]
                    
                    q_values[a] += prob * (reward + future_value) # Update q_values for action a in state s

            # Value of state is max over actions
            V[s] = np.max(q_values)
            delta = max(delta, abs(v - V[s]))
            
        # Extract the greedy policy from the current value function
        policy = np.zeros((env.n_states, env.n_actions))
        for s in range(env.n_states):
            q_values = np.zeros(env.n_actions)
            for a in range(env.n_actions):
                for prob, next_s, reward, done in env.get_transition_prob(s, a):
                    
                    if done:
                        future_val = 0  
                    else:
                        future_val = gamma * V[next_s]
                    
                    q_values[a] += prob * (reward + future_val) # Update q_values for action a in state s
            
            best_action = np.argmax(q_values) # Best action for state s
            policy[s, best_action] = 1.0 # Set the best action in the policy with probability 1
            
        yield {
            "values": V.copy(),
            "policy": policy,
            "iteration": i + 1,
            "delta": delta,
            "converged": delta < theta
        }
        
        if delta < theta: # If the value function has converged, break
            break