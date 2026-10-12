#!/usr/bin/env python3
"""Expectation Maximization for a GMM"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5,
                             verbose=False):
    """Perform the expectation maximization for a GMM

    Returns: pi, m, S, g, lik or None, None, None, None, None on failure
    """
    fail = (None, None, None, None, None)
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return fail
    if not isinstance(k, int) or k <= 0 or X.shape[0] < k:
        return fail
    if not isinstance(iterations, int) or iterations <= 0:
        return fail
    if not isinstance(tol, (int, float)) or tol < 0:
        return fail
    if not isinstance(verbose, bool):
        return fail
    pi, m, S = initialize(X, k)
    g, lik = expectation(X, pi, m, S)
    prev = lik
    i = 0
    for i in range(iterations):
        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}".format(
                i, round(lik, 5)))
        pi, m, S = maximization(X, g)
        g, lik = expectation(X, pi, m, S)
        if abs(lik - prev) <= tol:
            i += 1
            break
        prev = lik
    else:
        i = iterations
    if verbose:
        print("Log Likelihood after {} iterations: {}".format(
            i, round(lik, 5)))
    return pi, m, S, g, lik
