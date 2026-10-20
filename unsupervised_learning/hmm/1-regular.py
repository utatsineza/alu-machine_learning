#!/usr/bin/env python3
"""Steady state of a regular markov chain"""
import numpy as np


def regular(P):
    """Determine the steady state probabilities of a regular markov chain

    P: numpy.ndarray (n, n) transition matrix
    Returns: numpy.ndarray (1, n) or None on failure
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    n = P.shape[0]
    if P.shape != (n, n) or not np.allclose(np.sum(P, axis=1), 1):
        return None
    if not np.all(np.linalg.matrix_power(P, (n - 1) ** 2 + 1) > 0):
        return None
    w, v = np.linalg.eig(P.T)
    vec = np.real(v[:, np.argmin(np.abs(w - 1))])
    return (vec / np.sum(vec)).reshape(1, n)
