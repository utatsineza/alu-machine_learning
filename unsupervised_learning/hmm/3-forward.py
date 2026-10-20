#!/usr/bin/env python3
"""The forward algorithm for a hidden markov model"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Perform the forward algorithm for a hidden markov model

    Returns: P, F or None, None on failure
    P: likelihood of the observations given the model
    F: numpy.ndarray (N, T) of forward path probabilities
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
    F = np.zeros((N, T))
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
    for t in range(1, T):
        F[:, t] = (np.matmul(F[:, t - 1], Transition) *
                   Emission[:, Observation[t]])
    return np.sum(F[:, -1]), F
