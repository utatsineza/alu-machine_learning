#!/usr/bin/env python3
"""
Calculates the accuracy of a prediction
"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of the network's predictions

    Args:
        y: placeholder for the true labels (one-hot encoded)
        y_pred: tensor containing the network predictions

    Returns:
        A tensor containing the decimal accuracy
    """
    correct_predictions = tf.equal(
        tf.argmax(y, axis=1),
        tf.argmax(y_pred, axis=1)
    )

    accuracy = tf.reduce_mean(
        tf.cast(correct_predictions, tf.float32)
    )

    return accuracy
