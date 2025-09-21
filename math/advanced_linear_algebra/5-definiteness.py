#!/usr/bin/env python3
"""
Module to determine the definiteness of a numpy matrix.
"""

import numpy as np


def definiteness(matrix):
    """Determine the definiteness of a square matrix.

    Args:
        matrix (numpy.ndarray): The matrix to check.

    Returns:
        str or None: "Positive definite", "Positive semi-definite",
        "Negative definite", "Negative semi-definite",
        "Indefinite", or None if invalid.

    Raises:
        TypeError: If matrix is not a numpy.ndarray.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Check if matrix is square
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    if matrix.size == 0:
        return None

    # Check if matrix is symmetric
    if not np.allclose(matrix, matrix.T):
        return None

    # Compute eigenvalues
    try:
        eigvals = np.linalg.eigvalsh(
            matrix
        )  # Use eigvalsh for symmetric/hermitian matrices
    except np.linalg.LinAlgError:
        return None

    pos = np.all(eigvals > 0)
    pos_semi = np.all(eigvals >= 0)
    neg = np.all(eigvals < 0)
    neg_semi = np.all(eigvals <= 0)

    if pos:
        return "Positive definite"
    if pos_semi and not pos:
        return "Positive semi-definite"
    if neg:
        return "Negative definite"
    if neg_semi and not neg:
        return "Negative semi-definite"
    if np.any(eigvals > 0) and np.any(eigvals < 0):
        return "Indefinite"

    return None
