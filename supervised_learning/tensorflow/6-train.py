#!/usr/bin/env python3
"""
Builds, trains, and saves a neural network classifier
"""

import tensorflow as tf

create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier

    Args:
        X_train: training input data
        Y_train: training labels
        X_valid: validation input data
        Y_valid: validation labels
        layer_sizes: list of number of nodes per layer
        activations: list of activation functions per layer
        alpha: learning rate
        iterations: number of training iterations
        save_path: path to save the model

    Returns:
        The path where the model was saved
    """
    nx = X_train.shape[1]
    classes = Y_train.shape[1]

    x, y = create_placeholders(nx, classes)
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(iterations + 1):
            if i == 0 or i % 100 == 0 or i == iterations:
                train_cost, train_acc = sess.run(
                    [loss, accuracy],
                    feed_dict={x: X_train, y: Y_train}
                )
                valid_cost, valid_acc = sess.run(
                    [loss, accuracy],
                    feed_dict={x: X_valid, y: Y_valid}
                )

                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(train_cost))
                print("\tTraining Accuracy: {}".format(train_acc))
                print("\tValidation Cost: {}".format(valid_cost))
                print("\tValidation Accuracy: {}".format(valid_acc))

            if i < iterations:
                sess.run(
                    train_op,
                    feed_dict={x: X_train, y: Y_train}
                )

        save_path = saver.save(sess, save_path)

    return save_path
