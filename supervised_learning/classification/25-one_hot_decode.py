#!/usr/bin/env python3
"""
Converts a one-hot encoded matrix into a vector of labels
"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Decodes a one-hot encoded matrix

    Parameters:
    one_hot (numpy.ndarray): shape (classes, m)

    Returns:
    numpy.ndarray of shape (m,) containing labels, or None on failure
    """

    if not isinstance(one_hot, np.ndarray):
        return None

    if one_hot.ndim != 2:
        return None

    return np.argmax(one_hot, axis=0)
