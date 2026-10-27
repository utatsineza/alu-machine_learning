#!/usr/bin/env python3
"""Monte Carlo algorithm"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """Perform the Monte Carlo algorithm

    env: openAI environment instance
    V: numpy.ndarray (s,) value estimate
    policy: function that takes a state and returns the next action
    episodes: total number of episodes to train over
    max_steps: maximum number of steps per episode
    alpha: learning rate
    gamma: discount rate
    Returns: V, the updated value estimate
    """
    for i in range(episodes):
        state = env.reset()
        episode = []
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append([state, reward])
            if done:
                break
            state = next_state
        episode = np.array(episode, dtype=int)
        G = 0
        for state, reward in episode[::-1]:
            G = reward + gamma * G
            if state not in episode[:i, 0]:
                V[state] = V[state] + alpha * (G - V[state])
    return V
