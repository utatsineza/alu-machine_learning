#!/usr/bin/env python3
"""
Creates a layer of a neural network
"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Creates a layer for a neural network

    Args:
        prev: tensor output of the previous layer
        n: number of nodes in the layer
        activation: activation function to use

    Returns:
        The tensor output of the layer
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG"
    )

    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        name="layer"
    )

    return layer(prev)
