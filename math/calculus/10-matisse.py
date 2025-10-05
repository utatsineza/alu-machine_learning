#!/usr/bin/env python3
def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial.

    Parameters:
    poly (list of int/float): Coefficients of the polynomial.

    Returns:
    list of int/float: Coefficients of the derivative.
        Returns [0] if derivative is 0.
        Returns None if input is invalid.
    """
    # Validate input
    if not isinstance(poly, list) or not all(isinstance(x, (int, float)) for x in poly):
        return None

    # Derivative of a constant polynomial
    if len(poly) == 0 or all(x == 0 for x in poly):
        return [0]

    derivative = []
    for i in range(1, len(poly)):
        coeff = poly[i] * i
        if isinstance(coeff, float) and coeff.is_integer():
            coeff = int(coeff)
        derivative.append(coeff)

    # If derivative is empty (constant polynomial), return [0]
    if not derivative:
        return [0]

    return derivative
