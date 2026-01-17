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
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)  # number of layers
        self.__cache = {}     # dictionary to hold activations
        self.__weights = {}   # dictionary to hold weights and biases

        for l in range(self.L):
            nodes = layers[l]
            prev_nodes = nx if l == 0 else layers[l - 1]

            # He initialization
            self.__weights[f"W{l + 1}"] = np.random.randn(nodes, prev_nodes) * np.sqrt(2 / prev_nodes)
            self.__weights[f"b{l + 1}"] = np.zeros((nodes, 1))

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights

    def forward_prop(self, X):
        """
        Performs forward propagation of the network.

        Parameters:
        X (numpy.ndarray): shape (nx, m) input data

        Returns:
        A (numpy.ndarray): output of the network
        cache (dict): dictionary of all activations
        """
        self.__cache["A0"] = X
        A_prev = X

        for l in range(1, self.L + 1):
            W = self.__weights[f"W{l}"]
            b = self.__weights[f"b{l}"]

            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))  # sigmoid activation

            self.__cache[f"A{l}"] = A
            A_prev = A

        return A, self.__cache
