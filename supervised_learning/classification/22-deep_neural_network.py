#!/usr/bin/env python3
"""22-deep_neural_network.py"""

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

        self.L = len(layers)  # Number of layers
        self.cache = {}       # To store forward propagation outputs
        self.weights = {}     # To store weights and biases

        # --- Weight initialization (He et al.) ---
        layer_dims = [nx] + layers
        for l in range(1, self.L + 1):  # 1st loop
            self.weights[f"W{l}"] = (np.random.randn(layers[l - 1], layer_dims[l - 1])
                                      * np.sqrt(2 / layer_dims[l - 1]))
            self.weights[f"b{l}"] = np.zeros((layers[l - 1], 1))

    def forward_prop(self, X):
        """
        Performs forward propagation of the neural network.

        X: numpy.ndarray with shape (nx, m)
        Returns: output of the network and cache
        """
        self.cache["A0"] = X
        A_prev = X

        for l in range(1, self.L + 1):  # 2nd loop
            W = self.weights[f"W{l}"]
            b = self.weights[f"b{l}"]
            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))  # Sigmoid activation
            self.cache[f"A{l}"] = A
            A_prev = A

        return A, self.cache

    def evaluate(self, X, Y):
        """
        Evaluates the predictions of the network.

        X: numpy.ndarray of shape (nx, m)
        Y: numpy.ndarray of shape (1, m)
        Returns: prediction, cost
        """
        A, _ = self.forward_prop(X)
        cost = - (1 / Y.shape[1]) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the deep neural network.

        X: input data, shape (nx, m)
        Y: correct labels, shape (1, m)
        iterations: number of iterations
        alpha: learning rate
        Returns: evaluation of training data after iterations
        """
        # Exception checks
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        m = Y.shape[1]

        for i in range(iterations):  # 3rd loop: training iterations
            # Forward propagation
            A, _ = self.forward_prop(X)

            # Backpropagation through layers
            dZ = A - Y
            for l in reversed(range(1, self.L + 1)):  # 4th loop: layers
                A_prev = self.cache[f"A{l - 1}"]
                W = self.weights[f"W{l}"]
                dW = (1 / m) * np.matmul(dZ, A_prev.T)
                db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
                if l > 1:
                    A_prev_sig = self.cache[f"A{l - 1}"]
                    dZ = np.matmul(W.T, dZ) * A_prev_sig * (1 - A_prev_sig)
                # Update weights and biases
                self.weights[f"W{l}"] -= alpha * dW
                self.weights[f"b{l}"] -= alpha * db

        return self.evaluate(X, Y)
