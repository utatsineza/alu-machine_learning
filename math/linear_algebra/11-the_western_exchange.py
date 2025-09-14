#!/usr/bin/env python3
import numpy as np

def np_transpose(matrix: np.ndarray) -> np.ndarray:
    """
    Return the transpose of a NumPy array.

    Args:
        matrix (numpy.ndarray): Input array of any shape.

    Returns:
        numpy.ndarray: Transposed array.
            - For 1D arrays, returns as a 2D column vector.
            - For 2D arrays, returns the usual transpose.
            - For N-D arrays, reverses the order of axes.

    Examples:
        >>> np_transpose(np.array([1, 2, 3]))
        array([[1],
               [2],
               [3]])

        >>> np_transpose(np.array([[1, 2], [3, 4]]))
        array([[1, 3],
               [2, 4]])

        >>> np_transpose(np.array([[[1, 2]], [[3, 4]]])).shape
        (2, 1, 2)
    """
    if matrix.ndim == 1:
        return matrix.reshape(-1, 1)  # column vector
    return matrix.T

