#!/usr/bin/env python3
"""
Normal Distribution Class
This module defines a class that represents a normal distribution.
"""


class Normal:
    """
    Represents a normal (Gaussian) distribution.
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initializes a Normal distribution.

        Parameters:
        - data (list): the data used to estimate the distribution
        - mean (float): the mean of the distribution
        - stddev (float): the standard deviation of the distribution

        If data is provided, mean and stddev are calculated from it.
        Otherwise, mean and stddev are taken as given.
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Compute mean
            self.mean = float(sum(data) / len(data))

            # Compute standard deviation
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value.

        Parameters:
        - x (float): the x-value

        Returns:
        - z-score of x
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score.

        Parameters:
        - z (float): the z-score

        Returns:
        - x-value corresponding to z
        """
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value.

        Parameters:
        - x (float): the x-value

        Returns:
        - PDF value for x
        """
        e = 2.7182818285
        pi = 3.1415926536

        coeff = 1 / (self.stddev * ((2 * pi) ** 0.5))
        exponent = -0.5 * ((x - self.mean) / self.stddev) ** 2
        return coeff * (e ** exponent)

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value.

        Parameters:
        - x (float): the x-value

        Returns:
        - CDF value for x
        """
        # Approximation of the error function (erf)
        def erf(z):
            t = z
            return (2 / (3.1415926536 ** 0.5)) * (
                t - (t ** 3) / 3 + (t ** 5) / 10
                - (t ** 7) / 42 + (t ** 9) / 216
            )

        z = (x - self.mean) / (self.stddev * (2 ** 0.5))
        return 0.5 * (1 + erf(z))
