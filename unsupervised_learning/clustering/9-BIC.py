#!/usr/bin/env python3
"""Find the best number of clusters for a GMM using BIC"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Find the best number of clusters for a GMM using BIC

    Returns: best_k, best_result, l, b or None, None, None, None
    """
    fail = (None, None, None, None)
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return fail
    n, d = X.shape
    if kmax is None:
        kmax = n
    if not isinstance(kmin, int) or not isinstance(kmax, int):
        return fail
    if kmin <= 0 or kmax < kmin or kmax > n:
        return fail
    if not isinstance(iterations, int) or iterations <= 0:
        return fail
    if not isinstance(tol, (int, float)) or tol < 0:
        return fail
    if not isinstance(verbose, bool):
        return fail
    ls = []
    bs = []
    results = []
    for k in range(kmin, kmax + 1):
        pi, m, S, g, lik = expectation_maximization(
            X, k, iterations, tol, verbose)
        p = k * d + k * d * (d + 1) / 2 + k - 1
        ls.append(lik)
        bs.append(p * np.log(n) - 2 * lik)
        results.append((pi, m, S))
    ls = np.array(ls)
    bs = np.array(bs)
    best = np.argmin(bs)
    return best + kmin, results[best], ls, bs
