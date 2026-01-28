#!/usr/bin/env python3
"""
Creates the forward propagation graph for a neural network
"""

import tensorflow as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph

    Args:
        x: placeholder for input data
        layer_sizes: list of number of nodes in each layer
        activations: list of activation functions for each layer

    Returns:
        The prediction of the network in tensor form
    """
    output = x

    for i in range(len(layer_sizes)):
        with tf.variable_scope("layer_{}".format(i)):
            output = create_layer(
                output,
                layer_sizes[i],
                activations[i]
            )

    return output
