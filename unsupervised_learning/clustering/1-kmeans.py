#!/usr/bin/env python3
"""K-means clustering"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """Perform K-means on a dataset

    X: numpy.ndarray (n, d) dataset
    k: positive integer, number of clusters
    iterations: positive integer, maximum iterations
    Returns: C, clss or None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or X.shape[0] < k:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    low = X.min(axis=0)
    high = X.max(axis=0)
    C = np.random.uniform(low, high, size=(k, X.shape[1]))
    for _ in range(iterations):
        clss = np.argmin(np.linalg.norm(X[:, None] - C, axis=2), axis=1)
        new_C = C.copy()
        for j in range(k):
            pts = X[clss == j]
            if pts.shape[0] == 0:
                new_C[j] = np.random.uniform(low, high)
            else:
                new_C[j] = pts.mean(axis=0)
        if np.array_equal(new_C, C):
            break
        C = new_C
    clss = np.argmin(np.linalg.norm(X[:, None] - C, axis=2), axis=1)
    return C, clss
