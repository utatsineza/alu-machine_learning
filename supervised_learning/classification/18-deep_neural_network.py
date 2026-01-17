#!/usr/bin/env python3
"""18-deep_neural_network.py"""

import numpy as np


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing binary classification.
    """

    def __init__(self, nx, layers):
        """
        Class constructor

        nx: number of input features
        layers: list of number of nodes in each layer
        """
        # Input validation
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)  # number of layers
        self.cache = {}       # store forward propagation outputs
        self.weights = {}     # store weights and biases

        # He initialization for weights
        layer_dims = [nx] + layers
        for l in range(1, self.L + 1):  # loop allowed for initialization
            self.weights[f"W{l}"] = (np.random.randn(layers[l - 1], layer_dims[l - 1])
                                      * np.sqrt(2 / layer_dims[l - 1]))
            self.weights[f"b{l}"] = np.zeros((layers[l - 1], 1))

    def forward_prop(self, X):
        """
        Performs forward propagation through the network.

        X: numpy.ndarray of shape (nx, m)
        Returns: output of the network (A_L), cache
        """
        self.cache["A0"] = X
        A_prev = X

        for l in range(1, self.L + 1):  # only loop allowed
            W = self.weights[f"W{l}"]
            b = self.weights[f"b{l}"]
            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))  # Sigmoid activation
            self.cache[f"A{l}"] = A
            A_prev = A

        return self.cache[f"A{self.L}"], self.cache
