#!/usr/bin/env python3
"""The backward algorithm for a hidden markov model"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """Perform the backward algorithm for a hidden markov model

    Returns: P, B or None, None on failure
    P: likelihood of the observations given the model
    B: numpy.ndarray (N, T) of backward path probabilities
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
    B = np.ones((N, T))
    for t in range(T - 2, -1, -1):
        B[:, t] = np.matmul(
            Transition, Emission[:, Observation[t + 1]] * B[:, t + 1])
    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])
    return P, B
