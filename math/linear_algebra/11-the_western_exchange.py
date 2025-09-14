#!/usr/bin/env python3
"""
This module provides a function to transpose NumPy arrays.

The function np_transpose returns the transpose of an input array,
working with 1D, 2D, and higher-dimensional arrays.
"""


def np_transpose(matrix):
    """
    Transpose a NumPy array.

    Parameters
    ----------
    matrix : numpy.ndarray
        Input array of any shape.

    Returns
    -------
    numpy.ndarray
        Transposed array of the input.

    Examples
    --------
    >>> import numpy as np
    >>> np_transpose(np.array([[1, 2], [3, 4]]))
    array([[1, 3],
           [2, 4]])
    >>> np_transpose(np.array([1, 2, 3]))
    array([1, 2, 3])
    """
    return matrix.T

