#!/usr/bin/env python3
import numpy as np

def np_matmul(mat1, mat2):
    """
    Perform matrix multiplication of two numpy arrays.

    Args:
        mat1 (numpy.ndarray): First matrix.
        mat2 (numpy.ndarray): Second matrix.

    Returns:
        numpy.ndarray: Result of mat1 @ mat2.
    """
    return np.matmul(mat1, mat2)

