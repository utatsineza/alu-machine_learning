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
        C (int/float, optional): Constant of integration (default 0).

    Returns:
        list of int/float: Coefficients of the integrated polynomial.
        Returns None if input is invalid.
    """
    if not isinstance(poly, list) or not all(
        isinstance(x, (int, float)) for x in poly
    ):
        return None
    if not isinstance(C, (int, float)):
        return None
    if len(poly) == 0:
        return None

    result = [C]
    for i, coeff in enumerate(poly):
        new_coeff = coeff / (i + 1)
        if isinstance(new_coeff, float) and new_coeff.is_integer():
            new_coeff = int(new_coeff)
        result.append(new_coeff)

    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result
