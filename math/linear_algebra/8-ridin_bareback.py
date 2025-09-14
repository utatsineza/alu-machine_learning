#!/usr/bin/env python3
"""
This module provides a function for matrix multiplication (pure Python).
"""

def mat_mul(mat1, mat2):
    """
    Multiply two matrices.

    Parameters
    ----------
    mat1 : list of lists
    mat2 : list of lists

    Returns
    -------
    list of lists
        Product of mat1 and mat2, or None if not compatible.
    """
    if len(mat1[0]) != len(mat2):
        return None
    return [[sum(a * b for a, b in zip(r, c)) for c in zip(*mat2)] for r in mat1]

