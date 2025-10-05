#!/usr/bin/env python3
"""
Calculates the integral of a polynomial.
Contains the function poly_integral.
"""


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial.

    Parameters:
        poly (list of int/float): Coefficients of the polynomial.
            The index represents the power of x.
        C (int/float): Constant of integration.

    Returns:
        list of int/float: Coefficients of the integral.
            Returns None if input is invalid.
    """
    # Validate input
    if not isinstance(poly, list) or not all(isinstance(x, (int, float)) for x in poly):
        return None
    if not isinstance(C, (int, float)):
        return None

    # Compute integral coefficients
    integral = [C]
    for i in range(len(poly)):
        coeff = poly[i] / (i + 1)
        if coeff.is_integer():
            coeff = int(coeff)
        integral.append(coeff)

    # Remove trailing zeros if not needed
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral

