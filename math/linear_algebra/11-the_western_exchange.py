#!/usr/bin/env python3
def np_transpose(matrix):
    """
    Return the transpose of a NumPy array.

    Args:
        matrix (numpy.ndarray): Input array of any shape.

    Returns:
        numpy.ndarray: The transposed array.

    Examples:
        >>> import numpy as np
        >>> np_transpose(np.array([[1, 2], [3, 4]]))
        array([[1, 3],
               [2, 4]])
        >>> np_transpose(np.array([1, 2, 3]))
        array([1, 2, 3])
    """
    return matrix.T

