#!/usr/bin/env python3
import numpy as np

def np_elementwise(mat1, mat2):
    """
    Perform element-wise addition, subtraction, multiplication, and division
    on two numpy arrays (or an array and a scalar).

    Args:
        mat1 (numpy.ndarray): First input array.
        mat2 (numpy.ndarray or scalar): Second input array or scalar.

    Returns:
        tuple: (sum, difference, product, quotient)
    """
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2  # division always returns float in NumPy

    return add, sub, mul, div

