#!/usr/bin/env python3
"""
Defines a Deep Neural Network class for binary classification
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Deep Neural Network performing binary classification
    """

    def __init__(self, nx, layers):
        """
        Initializes the deep neural network

        Parameters:
        nx (int): number of input features
        layers (list): number of nodes in each layer
        """

        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate layers
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        # Private attributes
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # Initialize weights and biases (single loop)
        for l in range(1, self.__L + 1):
            nodes = layers[l - 1]
            if not isinstance(nodes, int) or nodes < 1:
                raise TypeError("layers must be a list of positive integers")

            prev_nodes = nx if l == 1 else layers[l - 2]

            self.__weights["W{}".format(l)] = (
                np.random.randn(nodes, prev_nodes) * np.sqrt(2 / prev_nodes)
            )
            self.__weights["b{}".format(l)] = np.zeros((nodes, 1))

    @property
    def L(self):
        """Returns the number of layers in the network"""
        return self.__L

    @property
    def cache(self):
        """Returns the cache dictionary"""
        return self.__cache

    @property
    def weights(self):
        """Returns the weights dictionary"""
        return self.__weights
