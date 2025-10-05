#!/usr/bin/env python3
def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial.

    Parameters:
    poly (list of int/float): List of coefficients representing a polynomial.
        The index of each element represents the power of x for that coefficient.
        Example: [5, 3, 0, 1] represents f(x) = x^3 + 3x + 5.
    C (int/float, optional): Constant of integration. Default is 0.

    Returns:
    list of int/float: Coefficients of the integrated polynomial, including C as
        the first element. Whole numbers are converted to int.
        Returns None if input is invalid.
    """
    # Validate inputs
    if not isinstance(poly, list) or not all(
            isinstance(x, (int, float)) for x in poly):
        return None
    if not isinstance(C, (int, float)):
        return None

    # Start with the constant of integration
    result = [C]

    # Integrate each term: a*x^n → a/(n+1)*x^(n+1)
    for i, coeff in enumerate(poly):
        new_coeff = coeff / (i + 1)
        if isinstance(new_coeff, float) and new_coeff.is_integer():
            new_coeff = int(new_coeff)
        result.append(new_coeff)

    # Remove unnecessary trailing zeros
    while len(result) > 1 and result[-1] == 0:
        result.pop()

    return result
