#!/usr/bin/env python3
"""
Evaluates the output of a neural network
"""

import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    Evaluates a trained neural network

    Args:
        X: numpy.ndarray containing the input data
        Y: numpy.ndarray containing the one-hot labels
        save_path: path to load the trained model from

    Returns:
        The network's prediction, accuracy, and loss
    """
    saver = tf.train.import_meta_graph(save_path + ".meta")

    with tf.Session() as sess:
        saver.restore(sess, save_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]

        pred, acc, cost = sess.run(
            [y_pred, accuracy, loss],
            feed_dict={x: X, y: Y}
        )

    return pred, acc, cost
