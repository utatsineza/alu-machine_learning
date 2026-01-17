#!/usr/bin/env python3
import numpy as np


class NeuralNetwork:
    """Defines a neural network with one hidden layer performing binary classification"""

    def __init__(self, nx, nodes):
        # Input validation
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Hidden layer
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        # Output neuron
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    # --- Getters for private attributes ---
    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    # --- Forward propagation ---
    def forward_prop(self, X):
        """Performs forward propagation"""
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))  # Sigmoid activation
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid activation
        return self.__A1, self.__A2

    # --- Cost computation ---
    def cost(self, Y, A):
        """Calculates the cost using logistic regression"""
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    # --- Evaluate predictions ---
    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions
        Returns: prediction, cost
        """
        _, A2 = self.forward_prop(X)
        prediction = np.where(A2 >= 0.5, 1, 0)
        cost = self.cost(Y, A2)
        return prediction, cost
