#!/usr/bin/env python3
import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification."""

    def __init__(self, nx, layers):
        # Input validation
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for l in range(self.L):
            nodes = layers[l]
            prev_nodes = nx if l == 0 else layers[l - 1]
            self.__weights[f"W{l + 1}"] = np.random.randn(nodes, prev_nodes) * np.sqrt(2 / prev_nodes)
            self.__weights[f"b{l + 1}"] = np.zeros((nodes, 1))

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """Performs forward propagation of the network"""
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

    def cost(self, Y, A):
        """Calculates cost using logistic regression"""
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the network’s predictions.
        Returns: prediction, cost
        """
        A, _ = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost
