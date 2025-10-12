#!/usr/bin/env python3
"""
Binomial Distribution Class
This module defines a class that represents a binomial distribution.
"""


class Binomial:
    """
    Represents a Binomial distribution.
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes a Binomial distribution.

        Parameters:
        - data (list): the data used to estimate the distribution
        - n (int): number of Bernoulli trials
        - p (float): probability of success

        If data is provided, n and p are estimated from it.
        Otherwise, n and p are taken as given.
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate sample mean and variance
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            # Estimate p first, then n
            p_est = 1 - (variance / mean)
            n_est = round(mean / p_est)

            # Recalculate p using estimated n
            p_est = mean / n_est

            self.n = int(n_est)
            self.p = float(p_est)

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of successes.

        Parameters:
        - k (int): number of successes

        Returns:
        - PMF value for k
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0 or k > self.n:
            return 0

        # Compute factorial without math library
        def factorial(num):
            result = 1
            for i in range(1, num + 1):
                result *= i
            return result

        nCk = factorial(self.n) / (factorial(k) * factorial(self.n - k))
        pmf = nCk * (self.p ** k) * ((1 - self.p) ** (self.n - k))
        return pmf

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of successes.

        Parameters:
        - k (int): number of successes

        Returns:
        - CDF value for k
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        if k > self.n:
            k = self.n

        cdf = 0
        for i in range(0, k + 1):
            cdf += self.pmf(i)
        return cdf
