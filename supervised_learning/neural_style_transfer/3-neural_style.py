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
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image so that:
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

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the Gram matrix of a layer
        """

        if (not isinstance(input_layer, (tf.Tensor, tf.Variable)) or
                len(input_layer.shape) != 4):
            raise TypeError(
                "input_layer must be a tensor of rank 4"
            )

        _, h, w, c = input_layer.shape

        gram = tf.linalg.einsum(
            'bijc,bijd->bcd',
            input_layer,
            input_layer
        )

        gram /= tf.cast(h * w, tf.float32)

        return gram

    def generate_features(self):
        """
        Extracts style and content features
        """

        # preprocess images for VGG19
        style_image = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255
        )

        content_image = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255
        )

        # get outputs
        style_outputs = self.model(style_image)
        content_outputs = self.model(content_image)

        # style features
        self.gram_style_features = [
            self.gram_matrix(style_output)
            for style_output in style_outputs[:-1]
        ]

        # content feature
        self.content_feature = content_outputs[-1]