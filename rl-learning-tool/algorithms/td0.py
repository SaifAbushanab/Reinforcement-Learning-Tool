"""
TD(0) - Temporal Difference Learning

Updates value estimates after every step using bootstrapping.

Update Rule:
    V(s) ← V(s) + α * [R + γV(s') - V(s)]
                      └────────────────┘
                          TD Error
"""

import numpy as np


def td_prediction(env, policy=None, gamma=0.99, alpha=0.1, epsilon=0.1, 
                  n_episodes=200):
    """
    TD(0) Prediction - learn V(s) using one-step bootstrapping.
    
    Args:
        env: Environment
        policy: Policy to evaluate
        gamma: Discount factor
        alpha: Learning rate
        epsilon: Exploration rate
        n_episodes: Number of episodes
        
    Yields:
        Dict with values, episode info
    """
    n_states = env.n_states
    n_actions = env.n_actions
    
    V = np.zeros(n_states)
    episode_rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            if policy is not None:
                action = np.random.choice(n_actions, p=policy[state])
            else:
                action = np.random.choice(n_actions)
            
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            # TD(0) Update
            if done:
                td_target = reward
            else:
                td_target = reward + gamma * V[next_state]
            
            V[state] = V[state] + alpha * (td_target - V[state])
            state = next_state
        
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 10 == 0 or episode == 0:
            yield {
                "values": V.copy(),
                "episode": episode + 1,
                "episode_reward": episode_reward,
                "episode_rewards": episode_rewards.copy(),
            }
    
    return V
