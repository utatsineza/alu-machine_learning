#!/usr/bin/env python3
"""
This module provides a function to concatenate NumPy arrays.
"""

import numpy as np

def np_cat(mat1, mat2, axis=0):
    """
    Concatenate two NumPy arrays along a given axis.

    Parameters
    ----------
    mat1 : numpy.ndarray
    mat2 : numpy.ndarray
    axis : int

    Returns
    -------
    numpy.ndarray
        Concatenated array.
    """
    return np.concatenate((mat1, mat2), axis=axis)

