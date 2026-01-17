#!/usr/bin/env python3
"""
14-neural_network.py
Defines a neural network with one hidden layer performing binary classification
and implements a fully vectorized training method with optional verbose output
and graphing of the training cost.
"""

import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """
    Neural network with one hidden layer performing binary classification.
    """

    def __init__(self, nx, nodes):
        """
        Initialize the neural network.

        Args:
            nx (int): Number of input features.
            nodes (int): Number of nodes in the hidden layer.

        Raises:
            TypeError: If nx or nodes is not an integer.
            ValueError: If nx or nodes is less than 1.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
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

    # --- Getters ---
    @property
    def W1(self):
        """Getter for W1."""
        return self.__W1

    @property
    def b1(self):
        """Getter for b1."""
        return self.__b1

    @property
    def A1(self):
        """Getter for A1."""
        return self.__A1

    @property
    def W2(self):
        """Getter for W2."""
        return self.__W2

    @property
    def b2(self):
        """Getter for b2."""
        return self.__b2

    @property
    def A2(self):
        """Getter for A2."""
        return self.__A2

    # --- Forward propagation ---
    def forward_prop(self, X):
        """
        Perform forward propagation.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m)

        Returns:
            tuple: (A1, A2) activated outputs
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    # --- Cost ---
    def cost(self, Y, A):
        """
        Calculate logistic regression cost.

        Args:
            Y (numpy.ndarray): Correct labels (1, m)
            A (numpy.ndarray): Activated output (1, m)

        Returns:
            float: Logistic regression cost
        """
        m = Y.shape[1]
        return - (1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

    # --- Gradient Descent ---
    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Perform one pass of gradient descent to update weights and biases.

        Args:
            X (numpy.ndarray): Input data (nx, m)
            Y (numpy.ndarray): Correct labels (1, m)
            A1 (numpy.ndarray): Hidden layer activations
            A2 (numpy.ndarray): Output layer activations
            alpha (float): Learning rate
        """
        m = Y.shape[1]

        dZ2 = A2 - Y
        dW2 = (1 / m) * np.matmul(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.matmul(self.__W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = (1 / m) * np.matmul(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2

    # --- Evaluate ---
    def evaluate(self, X, Y):
        """
        Evaluate the neural network's predictions.

        Args:
            X (numpy.ndarray): Input data (nx, m)
            Y (numpy.ndarray): Correct labels (1, m)

        Returns:
            tuple: (Predictions, cost)
        """
        _, A2 = self.forward_prop(X)
        prediction = np.where(A2 >= 0.5, 1, 0)
        return prediction, self.cost(Y, A2)

    # --- Train ---
    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Train the neural network.

        Args:
            X (numpy.ndarray): Input data (nx, m)
            Y (numpy.ndarray): Correct labels (1, m)
            iterations (int): Number of training iterations
            alpha (float): Learning rate
            verbose (bool): If True, print cost every step iterations
            graph (bool): If True, plot cost after training
            step (int): Interval of iterations to print/plot

        Returns:
            tuple: Predictions, cost after training
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if (verbose or graph) and (not isinstance(step, int)):
            raise TypeError("step must be an integer")
        if (verbose or graph) and (step < 1 or step > iterations):
            raise ValueError("step must be positive and <= iterations")

        costs = []
        iter_steps = []

        # Vectorized training using np.arange
        for i in range(iterations + 1):
            A1, A2 = self.forward_prop(X)
            cost = self.cost(Y, A2)

            if verbose and (i % step == 0 or i == 0 or i == iterations):
                print(f"Cost after {i} iterations: {cost}")
            if graph and (i % step == 0 or i == 0 or i == iterations):
                costs.append(cost)
                iter_steps.append(i)

            if i < iterations:
                self.gradient_descent(X, Y, A1, A2, alpha)

        if graph:
            plt.plot(iter_steps, costs, 'b-')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
