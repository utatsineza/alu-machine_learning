#!/usr/bin/env python3
"""
Exponential Distribution Class
This module defines a class that represents an exponential distribution.
"""


class Exponential:
    """
    Represents an exponential distribution.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes an Exponential distribution.

        Parameters:
        - data (list): the data used to estimate the distribution
        - lambtha (float): the expected number of occurrences

        If data is provided, lambtha is calculated as 1 / mean(data).
        Otherwise, lambtha is taken as given.
        """
        if data is None:
            # Validate given lambtha
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            # Validate data type and length
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate lambtha = 1 / mean(data)
            mean = sum(data) / len(data)
            self.lambtha = float(1 / mean)

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period.

        Parameters:
        - x (float): the time period

        Returns:
        - PDF value for x
        """
        if x < 0:
            return 0

        e = 2.7182818285
        pdf = self.lambtha * (e ** (-self.lambtha * x))
        return pdf

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given time period.

        Parameters:
        - x (float): the time period

        Returns:
        - CDF value for x
        """
        if x < 0:
            return 0

        e = 2.7182818285
        cdf = 1 - (e ** (-self.lambtha * x))
        return cdf
