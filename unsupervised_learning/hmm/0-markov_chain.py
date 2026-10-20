#!/usr/bin/env python3
"""Markov chain state probabilities"""
import numpy as np


def markov_chain(P, s, t=1):
    """Probability of a markov chain being in each state after t iterations

    P: numpy.ndarray (n, n) transition matrix
    s: numpy.ndarray (1, n) starting probabilities
    t: number of iterations
    Returns: numpy.ndarray (1, n) or None on failure
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    n = P.shape[0]
    if P.shape != (n, n):
        return None
    if not isinstance(s, np.ndarray) or s.shape != (1, n):
        return None
    if not isinstance(t, int) or t < 0:
        return None
    return np.matmul(s, np.linalg.matrix_power(P, t))
