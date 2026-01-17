def train(self, X, Y, iterations=5000, alpha=0.05):
    """
    Trains the neural network

    X: numpy.ndarray of shape (nx, m) - input data
    Y: numpy.ndarray of shape (1, m) - correct labels
    iterations: number of iterations to train
    alpha: learning rate
    Returns: evaluation of training data after iterations
    """
    # Input validation in exact order
    if not isinstance(iterations, int):
        raise TypeError("iterations must be an integer")
    if iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if not isinstance(alpha, float):
        raise TypeError("alpha must be a float")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    # Training loop - only one loop allowed
    for i in range(iterations):
        self.forward_prop(X)
        self.gradient_descent(X, Y, alpha)

    # Evaluate and return after training
    return self.evaluate(X, Y)
