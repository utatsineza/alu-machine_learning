#!/usr/bin/env python3
def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial.

    Parameters:
    poly (list of int/float): List of coefficients representing a polynomial.
        The index of each element represents the power of x for that coefficient.
        For example, [5, 3, 0, 1] represents f(x) = x^3 + 3x + 5.
    C (int/float, optional): Constant of integration. Default is 0.

    Returns:
    list of int/float: Coefficients of the integrated polynomial, including C as the first element.
        Coefficients are converted to integers if they are whole numbers.
        Returns None if input is invalid.
    """
    if not isinstance(poly, list) or not all(isinstance(x, (int, float)) for x in poly):
        return None
    if not isinstance(C, (int, float)):
        return None

    result = [C]
    for i in range(len(poly)):
        coeff = poly[i] / (i + 1)
        if coeff.is_integer():
            coeff = int(coeff)
        result.append(coeff)

    while len(result) > 1 and result[-1] == 0:
        result.pop()

    return result
