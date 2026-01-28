#!/usr/bin/env python3
"""
Calculates the softmax cross-entropy loss of a prediction
"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    Calculates the loss of the network's predictions

    Args:
        y: placeholder for the true labels (one-hot encoded)
        y_pred: tensor containing the network predictions

    Returns:
        A tensor containing the loss
    """
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=y,
        logits=y_pred
    )

    return loss
