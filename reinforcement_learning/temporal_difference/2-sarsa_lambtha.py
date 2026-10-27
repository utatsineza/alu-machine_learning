#!/usr/bin/env python3
"""SARSA(lambda) algorithm"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Choose an action with the epsilon greedy policy

    Q: numpy.ndarray (s, a) Q table
    state: current state
    epsilon: exploration threshold
    Returns: the chosen action
    """
    if np.random.uniform() < epsilon:
        return np.random.randint(Q.shape[1])
    return np.argmax(Q[state])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """Perform SARSA(lambda)

    env: openAI environment instance
    Q: numpy.ndarray (s, a) Q table
    lambtha: eligibility trace factor
    episodes: total number of episodes to train over
    max_steps: maximum number of steps per episode
    alpha: learning rate
    gamma: discount rate
    epsilon: initial threshold for epsilon greedy
    min_epsilon: minimum value that epsilon should decay to
    epsilon_decay: decay rate for updating epsilon between episodes
    Returns: Q, the updated Q table
    """
    init_epsilon = epsilon
    for i in range(episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        E = np.zeros_like(Q)
        for _ in range(max_steps):
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)
            delta = (reward + gamma * Q[next_state, next_action] -
                     Q[state, action])
            E = E * lambtha * gamma
            E[state, action] += 1
            Q = Q + alpha * delta * E
            if done:
                break
            state, action = next_state, next_action
        epsilon = (min_epsilon + (init_epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * i))
    return Q
