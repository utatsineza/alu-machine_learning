#!/usr/bin/env python3
"""
This module provides a function to perform matrix multiplication with NumPy.
"""

import numpy as np

def np_matmul(mat1, mat2):
    """
    Perform matrix multiplication.

    Parameters
    ----------
    mat1 : numpy.ndarray
    mat2 : numpy.ndarray

    Returns
    -------
    numpy.ndarray
        Matrix product.
    """
    return np.matmul(mat1, mat2)

