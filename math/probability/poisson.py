#!/usr/bin/env python3
"""
Poisson Distribution Class
This module defines a class that represents a Poisson distribution.
"""


class Poisson:
    """
    Represents a Poisson distribution.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes a Poisson distribution.

        Parameters:
        - data (list): the data used to estimate the distribution
        - lambtha (float): the expected number of occurrences

        If data is provided, lambtha is calculated from it.
        Otherwise, lambtha is taken as given.
        """
        if data is None:
            # If no data is provided, validate lambtha directly
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            # Validate data type and length
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Compute lambtha from data
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of “successes”.

        Parameters:
        - k (int): number of “successes”

        Returns:
        - PMF value for k
        """
        # Ensure k is an integer
        if not isinstance(k, int):
            try:
                k = int(k)
            except Exception:
                return 0

        # PMF is 0 for negative k
        if k < 0:
            return 0

        # Compute factorial manually (no math.factorial)
        fact = 1
        for i in range(1, k + 1):
            fact *= i

        # Apply Poisson PMF formula: P(k) = e^-λ * λ^k / k!
        e = 2.7182818285
        pmf = (e ** (-self.lambtha)) * (self.lambtha ** k) / fact
        return pmf

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of “successes”.

        Parameters:
        - k (int): number of “successes”

        Returns:
        - CDF value for k
        """
        # Ensure k is an integer
        if not isinstance(k, int):
            try:
                k = int(k)
            except Exception:
                return 0

        # CDF is 0 for negative k
        if k < 0:
            return 0

        # Sum PMF values from 0 to k
        cdf = 0
        for i in range(0, k + 1):
            cdf += self.pmf(i)
        return cdf
