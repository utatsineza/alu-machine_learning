#!/usr/bin/env python3
"""PCA that keeps a fraction of the variance"""
import numpy as np


def pca(X, var=0.95):
    """Perform PCA on a dataset

    X: numpy.ndarray (n, d), every dimension has a mean of 0
    var: fraction of the variance the transformation should maintain
    Returns: weights matrix W of shape (d, nd)
    """
    _, s, vh = np.linalg.svd(X)
    cum = np.cumsum(s) / np.sum(s)
    nd = np.argmax(cum >= var) + 1
    return vh.T[:, :nd]
