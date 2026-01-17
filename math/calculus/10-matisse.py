#!/usr/bin/env python3
"""
Calculates the derivative of a polynomial.
Contains the function poly_derivative.
"""


def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial.

    Parameters:
        poly (list of int/float): Coefficients of the polynomial.
            The index represents the power of x.

    Returns:
        list of int/float: Coefficients of the derivative.
            Returns [0] if derivative is zero.
            Returns None if input is invalid.
    """
    # Validate input
    if not isinstance(poly, list):
        return None
    if not poly or not all(isinstance(x, (int, float)) for x in poly):
        return None

    # Derivative of constant polynomial
    if len(poly) == 1:
        return [0]

    derivative = []
    for i in range(1, len(poly)):
        coeff = poly[i] * i
        if isinstance(coeff, float) and coeff.is_integer():
            coeff = int(coeff)
        derivative.append(coeff)

    # Handle case where derivative is 0
    if all(c == 0 for c in derivative):
        return [0]

    return derivative
