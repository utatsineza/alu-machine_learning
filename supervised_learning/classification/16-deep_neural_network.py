#!/usr/bin/env python3
"""16-deep_neural_network.py"""

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
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate layers
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)  # number of layers
        self.cache = {}       # dictionary to store activations
        self.weights = {}     # dictionary to store weights and biases

        # He initialization for weights
        layer_dims = [nx] + layers
        for l in range(1, self.L + 1):  # only loop allowed
            self.weights[f"W{l}"] = (np.random.randn(layers[l - 1], layer_dims[l - 1])
                                      * np.sqrt(2 / layer_dims[l - 1]))
            self.weights[f"b{l}"] = np.zeros((layers[l - 1], 1))
