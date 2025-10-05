#!/usr/bin/env python3
def poly_derivative(poly):
    # Validate input
    if not isinstance(poly, list) or not all(isinstance(x, (int, float)) for x in poly):
        return None

    # If the polynomial is constant (length 0 or 1), derivative is 0
    if len(poly) <= 1:
        return [0]

    # Compute derivative
    derivative = []
    for i in range(1, len(poly)):
        coeff = poly[i] * i
        # Convert to int if whole number
        if isinstance(coeff, float) and coeff.is_integer():
            coeff = int(coeff)
        derivative.append(coeff)

    # If derivative is empty, return [0]
    if not derivative:
        return [0]

    return derivative

