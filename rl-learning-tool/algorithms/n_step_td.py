"""
n-step TD Prediction

Generalizes TD(0) and Monte Carlo by looking n steps ahead.
- n=1: TD(0)
- n=∞: Monte Carlo

n-step Return:
    G = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n V(S_{t+n})
"""

import numpy as np


def n_step_td(env, policy=None, gamma=0.99, alpha=0.1, epsilon=0.1, 
              n_steps=3, n_episodes=200):
    """
    n-step TD Prediction.
    
    Args:
        env: Environment
        policy: Policy to evaluate
        gamma: Discount factor
        alpha: Learning rate
        epsilon: Exploration rate
        n_steps: Number of steps to look ahead
        n_episodes: Number of episodes
        
    Yields:
        Dict with values, episode info
    """
    n_states = env.n_states
    n_actions = env.n_actions
    
    V = np.zeros(n_states)
    episode_rewards = []
    
    for episode in range(n_episodes):
        states = [env.reset()]
        rewards = [0]
        
        done = False
        T = float('inf')
        t = 0
        
        while True:
            if t < T:
                if policy is not None:
                    action = np.random.choice(n_actions, p=policy[states[t]])
                else:
                    action = np.random.choice(n_actions)
                
                next_state, reward, done, _ = env.step(action)
                states.append(next_state)
                rewards.append(reward)
                
                if done:
                    T = t + 1
            
            tau = t - n_steps + 1
            
            if tau >= 0:
                G = 0
                for i in range(tau + 1, min(tau + n_steps, T) + 1):
                    G += (gamma ** (i - tau - 1)) * rewards[i]
                
                if tau + n_steps < T:
                    G += (gamma ** n_steps) * V[states[tau + n_steps]]
                
                V[states[tau]] = V[states[tau]] + alpha * (G - V[states[tau]])
            
            if tau == T - 1:
                break
            
            t += 1
        
        episode_rewards.append(sum(rewards))
        
        if (episode + 1) % 10 == 0 or episode == 0:
            yield {
                "values": V.copy(),
                "episode": episode + 1,
                "episode_reward": sum(rewards),
                "episode_rewards": episode_rewards.copy(),
                "n_steps": n_steps,
            }
    
    return V
