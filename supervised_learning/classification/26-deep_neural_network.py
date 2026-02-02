#!/usr/bin/env python3
"""
Defines a Deep Neural Network class for binary classification
"""
import numpy as np
import pickle
import os


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
        """Performs forward propagation"""

        self.__cache["A0"] = X

        for l in range(1, self.__L + 1):
            W = self.__weights["W{}".format(l)]
            b = self.__weights["b{}".format(l)]
            A_prev = self.__cache["A{}".format(l - 1)]

            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache["A{}".format(l)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """Computes cost"""

        m = Y.shape[1]
        return -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )

    def evaluate(self, X, Y):
        """Evaluates the network"""

        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = (A >= 0.5).astype(int)

        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Performs gradient descent"""

        m = Y.shape[1]
        dZ = cache["A{}".format(self.__L)] - Y

        for l in range(self.__L, 0, -1):
            A_prev = cache["A{}".format(l - 1)]
            W = self.__weights["W{}".format(l)]

            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            if l > 1:
                A_prev_act = A_prev
                dZ = np.matmul(W.T, dZ) * A_prev_act * (1 - A_prev_act)

            self.__weights["W{}".format(l)] -= alpha * dW
            self.__weights["b{}".format(l)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the neural network"""

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)

            if verbose and i % step == 0:
                print("Cost after {} iterations: {}".format(i, cost))

            self.gradient_descent(Y, cache, alpha)

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance to a file using pickle
        """

        if not filename.endswith(".pkl"):
            filename += ".pkl"

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object
        """

        if not os.path.exists(filename):
            return None

        with open(filename, "rb") as f:
            return pickle.load(f)
