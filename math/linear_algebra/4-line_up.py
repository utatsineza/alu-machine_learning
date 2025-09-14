#!/usr/bin/env python3
"""
This module provides a function to add two arrays element-wise.

If the arrays are not the same length, the function returns None.
"""

def add_arrays(arr1, arr2):
    """
    Add two arrays element-wise.

    Parameters
    ----------
    arr1 : list
        First array of numbers.
    arr2 : list
        Second array of numbers.

    Returns
    -------
    list or None
        A new list containing the element-wise sums,
        or None if the arrays have different lengths.

    Examples
    --------
    >>> add_arrays([1, 2, 3], [4, 5, 6])
    [5, 7, 9]
    >>> add_arrays([1, 2], [1])
    None
    """
    if len(arr1) != len(arr2):
        return None
    return [a + b for a, b in zip(arr1, arr2)]

