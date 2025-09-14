#!/usr/bin/env python3
"""
This module provides a function to slice a NumPy array.
"""

import numpy as np

def np_slice(matrix, axes=None):
    """
    Slice a NumPy array along multiple axes.

    Parameters
    ----------
    matrix : numpy.ndarray
        The array to slice.
    axes : dict, optional
        Dictionary where the key is the axis (int) and the value is a tuple
        of slice arguments (start, stop, step). Missing axes default to (:).

    Returns
    -------
    numpy.ndarray
        The sliced array.
    """
    if axes is None:
        axes = {}
    slices = []
    for i in range(matrix.ndim):
        if i in axes:
            slices.append(slice(*axes[i]))
        else:
            slices.append(slice(None))
    return matrix[tuple(slices)]

