#!/usr/bin/env python3
def poly_integral(poly, C=0):
    # Validate inputs
    if not isinstance(poly, list) or not all(isinstance(x, (int, float)) for x in poly):
        return None
    if not isinstance(C, (int, float)):
        return None

    # Start with the constant of integration
    result = [C]

    # Integrate each term: a*x^n → a/(n+1)*x^(n+1)
    for i in range(len(poly)):
        coeff = poly[i] / (i + 1)
        # Convert to int if it’s a whole number
        if coeff.is_integer():
            coeff = int(coeff)
        result.append(coeff)

    # Remove unnecessary trailing zeros (to make list as small as possible)
    while len(result) > 1 and result[-1] == 0:
        result.pop()

    return result

