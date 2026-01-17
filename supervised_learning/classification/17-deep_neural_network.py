#!/usr/bin/env python3
"""
17-deep_neural_network.py
Defines a deep neural network performing binary classification
with private attributes and proper getters.
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Deep neural network performing binary classification.

    Private attributes:
        __L: Number of layers in the network.
        __cache: Dictionary to store intermediary values.
        __weights: Dictionary to store weights and biases.
    """

    def __init__(self, nx, layers):
        """
        Initialize the deep neural network.

        Args:
            nx (int): Number of input features.
            layers (list): List of positive integers representing nodes per layer.

        Raises:
            TypeError: If nx is not int, or layers is not a list of positive integers.
            ValueError: If nx < 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0 or not all(
                isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # One loop for initializing all weights and biases
        for l in range(self.__L):
            layer_size = layers[l]
            prev_size = nx if l == 0 else layers[l - 1]
            self.__weights['W' + str(l + 1)] = np.random.randn(
                layer_size, prev_size) * np.sqrt(2 / prev_size)
            self.__weights['b' + str(l + 1)] = np.zeros((layer_size, 1))

    @property
    def L(self):
        """Number of layers in the network."""
        return self.__L

    @property
    def cache(self):
        """Dictionary holding intermediary values of the network."""
        return self.__cache

    @property
    def weights(self):
        """Dictionary holding weights and biases of the network."""
        return self.__weights
