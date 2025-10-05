#!/usr/bin/env python3
def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial.

    Parameters:
    poly (list of int/float): List of coefficients representing a polynomial.
        The index of each element represents the power of x for that coefficient.
        For example, [5, 3, 0, 1] represents f(x) = x^3 + 3x + 5.

    Returns:
    list of int/float: Coefficients of the derivative polynomial.
        If the derivative is 0, returns [0].
        Returns None if input is invalid.
    """
    # Validate input
    if not isinstance(poly, list) or not all(isinstance(x, (int, float)) for x in poly):
        return None

    if len(poly) <= 1:
        return [0]

    derivative = []
    for i in range(1, len(poly)):
        coeff = poly[i] * i
        if isinstance(coeff, float) and coeff.is_integer():
            coeff = int(coeff)
        derivative.append(coeff)

    return derivative if derivative else [0]

