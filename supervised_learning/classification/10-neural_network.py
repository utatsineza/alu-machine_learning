#!/usr/bin/env python3
import numpy as np


class NeuralNetwork:
    """Neural network with one hidden layer performing binary classification."""

    def __init__(self, nx, nodes):
        """
        Initialize the neural network.

        Parameters:
        nx (int): number of input features
        nodes (int): number of nodes in the hidden layer

        Raises:
        TypeError: if nx or nodes is not an integer
        ValueError: if nx or nodes is less than 1
        """
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
        self.__A1 = 0  # initialized as 0, will store array after forward_prop

        # Output neuron
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0  # initialized as 0, will store array after forward_prop

    # --- Getters ---
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
        """
        Perform forward propagation.

        Parameters:
        X (numpy.ndarray): input data of shape (nx, m)

        Updates:
        __A1 and __A2

        Returns:
        tuple: (__A1, __A2)
        """
        Z1 = self.__W1 @ X + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))  # sigmoid activation

        Z2 = self.__W2 @ self.__A1 + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))  # sigmoid activation

        return self.__A1, self.__A2
