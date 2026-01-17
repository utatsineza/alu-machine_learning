#!/usr/bin/env python3
import numpy as np


class NeuralNetwork:
    """Neural network with one hidden layer performing binary classification"""

    def __init__(self, nx, nodes):
        """
        Constructor for NeuralNetwork

        nx: number of input features
        nodes: number of nodes in the hidden layer
        """
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
        self.__A1 = np.zeros((nodes, 1))

        # Output neuron
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Weights of hidden layer"""
        return self.__W1

    @property
    def b1(self):
        """Bias of hidden layer"""
        return self.__b1

    @property
    def A1(self):
        """Activated output of hidden layer"""
        return self.__A1

    @property
    def W2(self):
        """Weights of output neuron"""
        return self.__W2

    @property
    def b2(self):
        """Bias of output neuron"""
        return self.__b2

    @property
    def A2(self):
        """Activated output of output neuron"""
        return self.__A2

    def forward_prop(self, X):
        """
        Performs forward propagation
        Updates __A1 and __A2
        Returns: __A1, __A2 (the same objects, not copies)
        """
        self.__A1[:] = 1 / (1 + np.exp(-(self.__W1 @ X + self.__b1)))
        self.__A2 = 1 / (1 + np.exp(-(self.__W2 @ self.__A1 + self.__b2)))

        return self.__A1, self.__A2
