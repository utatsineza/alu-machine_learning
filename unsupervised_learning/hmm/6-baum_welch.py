#!/usr/bin/env python3
"""The Baum-Welch algorithm for a hidden markov model"""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """Perform the Baum-Welch algorithm for a hidden markov model

    Returns: the converged Transition, Emission or None, None on failure
    """
    if not isinstance(Observations, np.ndarray) or Observations.ndim != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    M, N = Emission.shape
    if not isinstance(Transition, np.ndarray) or Transition.shape != (M, M):
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.shape != (M, 1):
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    T = Observations.shape[0]
    Tr = Transition.copy()
    E = Emission.copy()
    for _ in range(iterations):
        alpha = np.zeros((M, T))
        alpha[:, 0] = Initial[:, 0] * E[:, Observations[0]]
        for t in range(1, T):
            alpha[:, t] = (np.matmul(alpha[:, t - 1], Tr) *
                           E[:, Observations[t]])
        beta = np.ones((M, T))
        for t in range(T - 2, -1, -1):
            beta[:, t] = np.matmul(
                Tr, E[:, Observations[t + 1]] * beta[:, t + 1])
        xi = (alpha[:, None, :-1] * Tr[:, :, None] *
              E[:, Observations[1:]][None, :, :] * beta[None, :, 1:])
        xi = xi / np.sum(xi, axis=(0, 1))
        gamma = np.sum(xi, axis=1)
        last = alpha[:, -1] * beta[:, -1]
        gamma = np.hstack((gamma, (last / np.sum(last))[:, None]))
        Tr = np.sum(xi, axis=2) / np.sum(gamma[:, :-1], axis=1)[:, None]
        for k in range(N):
            E[:, k] = np.sum(gamma[:, Observations == k], axis=1)
        E = E / np.sum(gamma, axis=1)[:, None]
    return Tr, E
