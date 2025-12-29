"""
Monte Carlo Methods

MC Prediction: Learn V(s) by averaging returns from episodes.
MC Control: Learn optimal Q(s,a) with epsilon-greedy exploration.

Update Rule:
    V(s) ← V(s) + α * (G - V(s))
    Q(s,a) ← Q(s,a) + α * (G - Q(s,a))
"""

import numpy as np

def mc_prediction(env, policy=None, gamma=0.99, alpha=0.1, epsilon=0.1, 
                  n_episodes=200, first_visit=True):
    """Monte Carlo Prediction - estimate V(s) from episodes."""
    n_states = env.n_states
    n_actions = env.n_actions
    
    V = np.zeros(n_states)
    episode_rewards = []
    
    for episode in range(n_episodes):
        states, rewards = _generate_episode(env, policy, epsilon, n_actions)
        episode_rewards.append(sum(rewards))
        
        G = 0
        visited = set()
        
        for t in range(len(states) - 1, -1, -1):
            G = gamma * G + rewards[t]
            state = states[t]
            
            if first_visit and state in visited:
                continue
            visited.add(state)
            
            V[state] = V[state] + alpha * (G - V[state])
        
        if (episode + 1) % 10 == 0 or episode == 0:
            yield {
                "values": V.copy(),
                "episode": episode + 1,
                "episode_reward": sum(rewards),
                "episode_rewards": episode_rewards.copy(),
            }
    
    return V


def mc_control(env, gamma=0.99, alpha=0.1, epsilon=0.1, 
               n_episodes=200, epsilon_decay=0.995, min_epsilon=0.01):
    """Monte Carlo Control - learn optimal Q(s,a)."""
    n_states = env.n_states
    n_actions = env.n_actions
    
    Q = np.zeros((n_states, n_actions))
    episode_rewards = []
    current_epsilon = epsilon
    
    for episode in range(n_episodes):
        states, actions, rewards = _generate_episode_q(env, Q, current_epsilon, n_actions)
        episode_rewards.append(sum(rewards))
        
        G = 0
        visited = set()
        
        for t in range(len(states) - 1, -1, -1):
            G = gamma * G + rewards[t]
            state, action = states[t], actions[t]
            
            if (state, action) in visited:
                continue
            visited.add((state, action))
            
            Q[state, action] = Q[state, action] + alpha * (G - Q[state, action])
        
        current_epsilon = max(min_epsilon, current_epsilon * epsilon_decay)
        
        policy = np.zeros((n_states, n_actions))
        for s in range(n_states):
            policy[s, np.argmax(Q[s])] = 1.0
        
        if (episode + 1) % 10 == 0 or episode == 0:
            yield {
                "Q": Q.copy(),
                "values": np.max(Q, axis=1),
                "policy": policy,
                "episode": episode + 1,
                "episode_reward": sum(rewards),
                "episode_rewards": episode_rewards.copy(),
                "epsilon": current_epsilon,
            }
    
    return Q


def _generate_episode(env, policy, epsilon, n_actions):
    """Generate episode for prediction."""
    states, rewards = [], []
    state = env.reset()
    done = False
    
    while not done and len(states) < 1000:
        states.append(state)
        if policy is not None:
            action = np.random.choice(n_actions, p=policy[state])
        else:
            action = np.random.choice(n_actions)
        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        state = next_state
    
    return states, rewards


def _generate_episode_q(env, Q, epsilon, n_actions):
    """Generate episode for control using epsilon-greedy."""
    states, actions, rewards = [], [], []
    state = env.reset()
    done = False
    
    while not done and len(states) < 1000:
        states.append(state)
        if np.random.random() < epsilon:
            action = np.random.choice(n_actions)
        else:
            action = np.argmax(Q[state])
        actions.append(action)
        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        state = next_state
    
    return states, actions, rewards
