#!/usr/bin/env python3
"""Neural Style Transfer module"""

import tensorflow as tf
import numpy as np


class NST:
    """Performs tasks for neural style transfer"""

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]

    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image,
                 alpha=1e4, beta=1):
        """
        Class constructor
        """

        if (not isinstance(style_image, np.ndarray) or
                len(style_image.shape) != 3 or
                style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (not isinstance(content_image, np.ndarray) or
                len(content_image.shape) != 3 or
                content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray "
                "with shape (h, w, 3)"
            )

        if (not isinstance(alpha, (int, float)) or alpha < 0):
            raise TypeError("alpha must be a non-negative number")

        if (not isinstance(beta, (int, float)) or beta < 0):
            raise TypeError("beta must be a non-negative number")

        tf.enable_eager_execution()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)

        self.alpha = alpha
        self.beta = beta

        self.load_model()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that:
        - largest side is 512 pixels
        - values are between 0 and 1
        """

        if (not isinstance(image, np.ndarray) or
                len(image.shape) != 3 or
                image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape

        if h > w:
            new_h = 512
            new_w = int(w * 512 / h)
        else:
            new_w = 512
            new_h = int(h * 512 / w)

        resized = tf.image.resize(
            image,
            (new_h, new_w),
            method=tf.image.ResizeMethod.BICUBIC
        )

        scaled = resized / 255.0

        scaled = tf.clip_by_value(scaled, 0.0, 1.0)

        return tf.expand_dims(scaled, axis=0)

    def load_model(self):
        """
        Creates the model used to calculate cost
        """

        vgg19 = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )

        vgg19.trainable = False

        outputs = []

        for layer_name in self.style_layers:
            outputs.append(vgg19.get_layer(layer_name).output)

        outputs.append(
            vgg19.get_layer(self.content_layer).output
        )

        self.model = tf.keras.models.Model(
            inputs=vgg19.input,
            outputs=outputs
        )

        self.model.trainable = False

        # Replace MaxPooling2D with AveragePooling2D
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                self.model.layers[i] = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding
                )