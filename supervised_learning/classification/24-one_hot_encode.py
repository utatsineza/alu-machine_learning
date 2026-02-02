#!/usr/bin/env python3
"""
Converts a numeric label vector into a one-hot encoded matrix
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    One-hot encodes a numeric label vector

    Parameters:
    Y (numpy.ndarray): shape (m,) containing labels
    classes (int): total number of classes

    Returns:
    numpy.ndarray of shape (classes, m) or None on failure
    """

    if not isinstance(Y, np.ndarray):
        return None

    if not isinstance(classes, int):
        return None

    if classes < 2:
        return None

    if Y.ndim != 1:
        return None

    if np.min(Y) < 0:
        return None

    if np.max(Y) >= classes:
        return None

    m = Y.shape[0]
    one_hot = np.zeros((classes, m))

    one_hot[Y, np.arange(m)] = 1

    return one_hot
