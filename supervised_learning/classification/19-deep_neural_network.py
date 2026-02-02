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
        """

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # Loop 1: initialize weights and biases
        for l in range(1, self.__L + 1):
            nodes = layers[l - 1]
            if not isinstance(nodes, int) or nodes < 1:
                raise TypeError("layers must be a list of positive integers")

            prev = nx if l == 1 else layers[l - 2]

            self.__weights["W{}".format(l)] = (
                np.random.randn(nodes, prev) * np.sqrt(2 / prev)
            )
            self.__weights["b{}".format(l)] = np.zeros((nodes, 1))

    @property
    def L(self):
        """Returns the number of layers"""
        return self.__L

    @property
    def cache(self):
        """Returns the cache"""
        return self.__cache

    @property
    def weights(self):
        """Returns the weights"""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates forward propagation
        """

        self.__cache["A0"] = X

        # Loop 2: forward propagation
        for l in range(1, self.__L + 1):
            W = self.__weights["W{}".format(l)]
            b = self.__weights["b{}".format(l)]
            A_prev = self.__cache["A{}".format(l - 1)]

            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))

            self.__cache["A{}".format(l)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost using logistic regression
        """

        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost
