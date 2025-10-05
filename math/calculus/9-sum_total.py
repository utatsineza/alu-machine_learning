#!/usr/bin/env python3
"""
Calculates the sum of squares of integers from 1 to n.
Contains the function summation_i_squared.
"""


def summation_i_squared(n):
    """
    Calculates the sum of squares from 1 to n without using loops.

    Parameters:
    n (int): The upper bound of the summation (must be a positive integer).

    Returns:
    int: The sum 1^2 + 2^2 + ... + n^2.
    None: If n is not a valid positive integer.
    """
    if not isinstance(n, int) or n < 1:
        return None

    return n * (n + 1) * (2 * n + 1) // 6
