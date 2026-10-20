#!/usr/bin/env python3
"""The Viterbi algorithm for a hidden markov model"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """Calculate the most likely sequence of hidden states

    Returns: path, P or None, None on failure
    path: list of length T with the most likely hidden states
    P: probability of obtaining the path sequence
    """
    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    N = Emission.shape[0]
    if not isinstance(Transition, np.ndarray) or Transition.shape != (N, N):
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.shape != (N, 1):
        return None, None
    T = Observation.shape[0]
    V = np.zeros((N, T))
    back = np.zeros((N, T), dtype=int)
    V[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
    for t in range(1, T):
        cand = V[:, t - 1, None] * Transition
        back[:, t] = np.argmax(cand, axis=0)
        V[:, t] = np.max(cand, axis=0) * Emission[:, Observation[t]]
    last = int(np.argmax(V[:, -1]))
    path = [last]
    for t in range(T - 1, 0, -1):
        last = int(back[last, t])
        path.append(last)
    return path[::-1], np.max(V[:, -1])
