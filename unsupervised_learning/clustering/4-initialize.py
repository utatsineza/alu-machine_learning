#!/usr/bin/env python3
"""Initialize a Gaussian Mixture Model"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initialize variables for a GMM

    Returns: pi, m, S or None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0 or X.shape[0] < k:
        return None, None, None
    d = X.shape[1]
    pi = np.full(k, 1 / k)
    m, _ = kmeans(X, k)
    S = np.tile(np.eye(d), (k, 1, 1))
    return pi, m, S
