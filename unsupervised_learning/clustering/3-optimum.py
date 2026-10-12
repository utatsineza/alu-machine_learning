#!/usr/bin/env python3
"""Optimize the number of clusters for K-means"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Test for the optimum number of clusters by variance

    Returns: results, d_vars or None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    n = X.shape[0]
    if kmax is None:
        kmax = n
    if not isinstance(kmin, int) or not isinstance(kmax, int):
        return None, None
    if kmin <= 0 or kmax <= 0 or kmax <= kmin or kmax > n:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    results = []
    variances = []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        variances.append(variance(X, C))
    d_vars = [variances[0] - v for v in variances]
    return results, d_vars
