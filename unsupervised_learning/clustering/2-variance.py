#!/usr/bin/env python3
"""Total intra-cluster variance"""
import numpy as np


def variance(X, C):
    """Calculate the total intra-cluster variance

    X: numpy.ndarray (n, d) dataset
    C: numpy.ndarray (k, d) centroid means
    Returns: var, or None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None
    dist = np.sum((X[:, None] - C) ** 2, axis=2)
    return np.sum(np.min(dist, axis=1))
