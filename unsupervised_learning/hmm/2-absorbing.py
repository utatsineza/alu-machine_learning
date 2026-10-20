#!/usr/bin/env python3
"""Absorbing markov chains"""
import numpy as np


def absorbing(P):
    """Determine if a markov chain is absorbing

    P: numpy.ndarray (n, n) standard transition matrix
    Returns: True if absorbing, else False
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return False
    n = P.shape[0]
    if P.shape != (n, n) or not np.allclose(np.sum(P, axis=1), 1):
        return False
    absorb = np.diag(P) == 1
    if not np.any(absorb):
        return False
    reach = np.linalg.matrix_power((P > 0).astype(int) + np.eye(n, dtype=int),
                                   n) > 0
    return bool(np.all(np.any(reach[:, absorb], axis=1)))
