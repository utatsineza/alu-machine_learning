#!/usr/bin/env python3
"""TD(lambda) algorithm"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """Perform the TD(lambda) algorithm

    env: openAI environment instance
    V: numpy.ndarray (s,) value estimate
    policy: function that takes a state and returns the next action
    lambtha: eligibility trace factor
    episodes: total number of episodes to train over
    max_steps: maximum number of steps per episode
    alpha: learning rate
    gamma: discount rate
    Returns: V, the updated value estimate
    """
    for _ in range(episodes):
        state = env.reset()
        E = np.zeros_like(V)
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            E = E * lambtha * gamma
            E[state] += 1
            delta = reward + gamma * V[next_state] - V[state]
            V = V + alpha * delta * E
            if done:
                break
            state = next_state
    return V
