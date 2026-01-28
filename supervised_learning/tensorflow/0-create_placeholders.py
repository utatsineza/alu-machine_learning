#!/usr/bin/env python3
"""
Creates placeholders for a neural network
"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Creates placeholders for the input data and labels

    Args:
        nx: number of feature columns
        classes: number of classes for classification

    Returns:
        x: placeholder for input data
        y: placeholder for one-hot encoded labels
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")

    return x, y
