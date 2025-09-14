#!/usr/bin/env python3
"""
This module provides element-wise arithmetic operations on NumPy arrays.
"""

def np_elementwise(mat1, mat2):
    """
    Perform element-wise addition, subtraction, multiplication, and division.

    Parameters
    ----------
    mat1 : numpy.ndarray
    mat2 : numpy.ndarray or scalar

    Returns
    -------
    tuple
        (sum, difference, product, quotient) of element-wise operations.
    """
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2

