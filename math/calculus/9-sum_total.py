#!/usr/bin/env python3
def summation_i_squared(n):
    # Validate input
    if not isinstance(n, int) or n < 1:
        return None

    # Use formula: n(n + 1)(2n + 1) / 6
    result = n * (n + 1) * (2 * n + 1) // 6
    return result

