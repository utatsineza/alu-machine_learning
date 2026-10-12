#!/usr/bin/env python3
"""PCA to a fixed number of dimensions"""
import numpy as np


def pca(X, ndim):
    """Perform PCA on a dataset

    X: numpy.ndarray (n, d) dataset
    ndim: new dimensionality of the transformed X
    Returns: T numpy.ndarray (n, ndim), the transformed X
    """
    X_m = X - np.mean(X, axis=0)
    _, _, vh = np.linalg.svd(X_m)
    return np.matmul(X_m, vh.T[:, :ndim])
