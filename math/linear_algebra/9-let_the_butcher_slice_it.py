#!/usr/bin/env python3
"""
This module provides a function to slice a matrix with NumPy.
"""

def np_slice(matrix, axes={}):
    """
    Slice a matrix along multiple axes.

    Parameters
    ----------
    matrix : numpy.ndarray
    axes : dict
        Dictionary of slices per axis.

    Returns
    -------
    numpy.ndarray
        The sliced matrix.
    """
    slices = [slice(*axes.get(i, (None, None, None))) for i in range(matrix.ndim)]
    return matrix[tuple(slices)]

