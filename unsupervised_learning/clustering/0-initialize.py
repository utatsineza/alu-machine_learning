#!/usr/bin/env python3
"""Initialize K-means centroids"""
import numpy as np


def initialize(X, k):
    """Initialize cluster centroids for K-means

    X: numpy.ndarray (n, d) dataset
    k: positive integer, number of clusters
    Returns: numpy.ndarray (k, d) of centroids, or None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None
    low = X.min(axis=0)
    high = X.max(axis=0)
    return np.random.uniform(low, high, size=(k, X.shape[1]))
