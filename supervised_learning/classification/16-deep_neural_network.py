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
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        for nodes in layers:
            if not isinstance(nodes, int) or nodes < 1:
                raise TypeError("layers must be a list of positive integers")

        # Public attributes
        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        # He initialization (only one loop)
        for l in range(1, self.L + 1):
            layer_size = layers[l - 1]
            prev_size = nx if l == 1 else layers[l - 2]

            self.weights["W{}".format(l)] = (
                np.random.randn(layer_size, prev_size) * np.sqrt(2 / prev_size)
            )
            self.weights["b{}".format(l)] = np.zeros((layer_size, 1))
