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

        Args:
            style_image: numpy.ndarray of shape (h, w, 3)
            content_image: numpy.ndarray of shape (h, w, 3)
            alpha: weight for content cost
            beta: weight for style cost
        """
        if (not isinstance(style_image, np.ndarray) or
                len(style_image.shape) != 3 or
                style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray "
                "with shape (h, w, 3)"
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

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixel values are between
        0 and 1 and its largest side is 512 pixels

        Args:
            image: numpy.ndarray of shape (h, w, 3)

        Returns:
            tf.Tensor of shape (1, h_new, w_new, 3)
        """
        if (not isinstance(image, np.ndarray) or
                len(image.shape) != 3 or
                image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray "
                "with shape (h, w, 3)"
            )

        h, w, _ = image.shape
        if h > w:
            new_h = 512
            new_w = int(w * 512 / h)
        else:
            new_w = 512
            new_h = int(h * 512 / w)

        image = tf.cast(image, tf.float32)
        resized = tf.image.resize(
            tf.expand_dims(image, axis=0),
            (new_h, new_w),
            method=tf.image.ResizeMethod.BICUBIC
        )
        scaled = resized / 255.0
        scaled = tf.clip_by_value(scaled, 0, 1)

        return scaled

    def load_model(self):
        """
        Creates the model used to calculate cost

        Returns:
            None
        """
        vgg19 = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )

        vgg19.trainable = False

        style_outputs = [
            vgg19.get_layer(layer).output
            for layer in self.style_layers
        ]

        content_output = vgg19.get_layer(self.content_layer).output
        outputs = style_outputs + [content_output]

        self.model = tf.keras.models.Model(
            inputs=vgg19.input,
            outputs=outputs
        )

        self.model.trainable = False

        