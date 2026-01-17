#!/usr/bin/env python3
"""
21-deep_neural_network.py
Defines a deep neural network performing binary classification
and implements gradient descent.
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

        for l in range(self.__L):
            layer_size = layers[l]
            prev_size = nx if l == 0 else layers[l - 1]
            self.__weights['W' + str(l + 1)] = (np.random.randn(layer_size, prev_size)
                                                * np.sqrt(2 / prev_size))
            self.__weights['b' + str(l + 1)] = np.zeros((layer_size, 1))

    @property
    def L(self):
        """Number of layers."""
        return self.__L

    @property
    def cache(self):
        """Dictionary holding intermediary values of the network."""
        return self.__cache

    @property
    def weights(self):
        """Dictionary holding weights and biases of the network."""
        return self.__weights

    def forward_prop(self, X):
        """
        Perform forward propagation of the deep neural network.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m)

        Returns:
            tuple: (output of last layer, cache dictionary)
        """
        self.__cache['A0'] = X
        for l in range(1, self.__L + 1):
            Wl = self.__weights['W' + str(l)]
            bl = self.__weights['b' + str(l)]
            Al_prev = self.__cache['A' + str(l - 1)]
            Zl = np.matmul(Wl, Al_prev) + bl
            self.__cache['A' + str(l)] = 1 / (1 + np.exp(-Zl))  # Sigmoid activation
        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Calculate the cost using logistic regression.

        Args:
            Y (numpy.ndarray): Correct labels of shape (1, m)
            A (numpy.ndarray): Activated output of the last layer (1, m)

        Returns:
            float: Logistic regression cost
        """
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Perform one pass of gradient descent on the deep neural network.

        Args:
            Y (numpy.ndarray): Correct labels of shape (1, m)
            cache (dict): Dictionary containing all intermediary values
            alpha (float): Learning rate
        """
        m = Y.shape[1]
        weights_copy = self.__weights.copy()
        L = self.__L
        dZ = {}  # Dictionary to hold delta values

        # Initialize delta for last layer
        A_last = cache['A' + str(L)]
        dZ[L] = A_last - Y

        for l in range(L, 0, -1):
            Al_prev = cache['A' + str(l - 1)]
            Wl = weights_copy['W' + str(l)]
            # Gradients
            dW = (1 / m) * np.matmul(dZ[l], Al_prev.T)
            db = (1 / m) * np.sum(dZ[l], axis=1, keepdims=True)
            # Update weights and biases
            self.__weights['W' + str(l)] -= alpha * dW
            self.__weights['b' + str(l)] -= alpha * db
            if l > 1:
                # Delta for previous layer
                Al_prev_sig = cache['A' + str(l - 1)]
                dZ[l - 1] = np.matmul(weights_copy['W' + str(l)].T, dZ[l]) * \
                             (Al_prev_sig * (1 - Al_prev_sig))
