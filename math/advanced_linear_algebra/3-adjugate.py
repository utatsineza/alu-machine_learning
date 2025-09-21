#!/usr/bin/env python3
"""
Module to calculate the adjugate matrix of a square matrix.
"""

from 2-cofactor import cofactor



def determinant(matrix):
    """Helper function to calculate the determinant of a square matrix."""
    """
    Calculate the adjugate matrix of a square matrix.

    Args:
        matrix (list of lists): The square matrix.

    Returns:
        list of lists: Adjugate matrix.
    """
    if matrix == [[]]:
        return 1
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    det = 0
    for c in range(n):
        submatrix = [row[:c] + row[c+1:] for row in matrix[1:]]
        det += ((-1) ** c) * matrix[0][c] * determinant(submatrix)
    return det


def minor(matrix):
    """Helper function to calculate the minor matrix of a square matrix."""
    n = len(matrix)
    if n == 1:
        return [[1]]

    minor_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            submatrix = [r[:j] + r[j+1:] for k, r in enumerate(matrix) if k != i]
            row.append(determinant(submatrix))
        minor_matrix.append(row)
    return minor_matrix


def cofactor(matrix):
    """Helper function to calculate the cofactor matrix."""
    n = len(matrix)
    minor_matrix = minor(matrix)
    cofactor_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(minor_matrix[i][j] * ((-1) ** (i + j)))
        cofactor_matrix.append(row)
    return cofactor_matrix


def adjugate(matrix):
    """
    Calculates the adjugate matrix of a square matrix.

    Args:
        matrix (list of lists): The matrix to calculate the adjugate matrix of.

    Returns:
        list of lists: The adjugate matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not square or is empty.
    """
    # Type check
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check non-empty and square
    n = len(matrix)
    if n == 0 or not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Adjugate is the transpose of the cofactor matrix
    cofactor_matrix = cofactor(matrix)
    adjugate_matrix = [[cofactor_matrix[j][i] for j in range(n)] for i in range(n)]

    return adjugate_matrix

