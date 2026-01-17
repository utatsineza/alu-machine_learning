#!/usr/bin/env python3
import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """
        nx: number of input features
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Public instance attributes
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
