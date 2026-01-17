#!/usr/bin/env python3
import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification."""

    def __init__(self, nx, layers):
        """
        Initialize the deep neural network.

        Parameters:
        nx (int): Number of input features.
        layers (list): Number of nodes in each layer.

        Raises:
        TypeError: if nx is not an integer.
        ValueError: if nx < 1.
        TypeError: if layers is not a list of positive integers.
        """
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate layers
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)        # Number of layers
        self.cache = {}             # To store activations
        self.weights = {}           # To store weights and biases

        # Initialize weights and biases using He et al. method
        for l in range(self.L):
            layer_size = layers[l]
            prev_size = nx if l == 0 else layers[l - 1]

            self.weights["W{}".format(l + 1)] = np.random.randn(layer_size, prev_size) * np.sqrt(2 / prev_size)
            self.weights["b{}".format(l + 1)] = np.zeros((layer_size, 1))
