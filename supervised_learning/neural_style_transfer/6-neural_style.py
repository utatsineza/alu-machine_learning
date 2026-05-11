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
        self.generate_features()

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

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the Gram matrix of a layer

        Args:
            input_layer: tensor of shape (1, h, w, c)

        Returns:
            Gram matrix as a tensor of shape (1, c, c)
        """
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable)) or
                len(input_layer.shape) != 4):
            raise TypeError(
                "input_layer must be a tensor of rank 4"
            )

        gram = tf.linalg.einsum(
            'bijc,bijd->bcd',
            input_layer,
            input_layer
        )

        input_shape = tf.shape(input_layer)
        h = input_shape[1]
        w = input_shape[2]
        gram /= tf.cast(h * w, tf.float32)

        return gram

    def generate_features(self):
        """
        Extracts style and content features from the images

        Returns:
            None
        """
        style_image = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255
        )
        content_image = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255
        )

        style_outputs = self.model(style_image)
        content_outputs = self.model(content_image)

        self.gram_style_features = [
            self.gram_matrix(output)
            for output in style_outputs[:-1]
        ]

        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer

        Args:
            style_output: tensor of shape (1, h, w, c)
            gram_target: tensor of shape (1, c, c)

        Returns:
            Style cost for the layer as a scalar tensor
        """
        if (not isinstance(style_output, (tf.Tensor, tf.Variable)) or
                len(style_output.shape) != 4):
            raise TypeError(
                "style_output must be a tensor of rank 4"
            )

        _, _, _, c = style_output.shape

        if (not isinstance(gram_target, (tf.Tensor, tf.Variable)) or
                gram_target.shape != (1, c, c)):
            raise TypeError(
                "gram_target must be a tensor of shape "
                "[1, {}, {}] where {} is the number "
                "of channels in style_output".format(c, c, c)
            )

        gram_style = self.gram_matrix(style_output)

        style_cost = tf.reduce_sum(
            tf.square(gram_style - gram_target)
        ) / tf.cast(c ** 2, tf.float32)

        return style_cost

    def style_cost(self, style_outputs):
        """
        Calculates the total style cost for the generated image

        Args:
            style_outputs: list of tf.Tensor style outputs
                for the generated image

        Returns:
            Total style cost as a scalar tensor
        """
        if (not isinstance(style_outputs, list) or
                len(style_outputs) != len(self.style_layers)):
            raise TypeError(
                "style_outputs must be a list with a length of {}".format(
                    len(self.style_layers)
                )
            )

        weight = 1 / len(self.style_layers)
        total_style_cost = 0

        for style_output, gram_target in zip(
            style_outputs, self.gram_style_features
        ):
            total_style_cost += weight * self.layer_style_cost(
                style_output, gram_target
            )

        return total_style_cost

    def content_cost(self, content_output):
        """
        Calculates the content cost for the generated image

        Args:
            content_output: tf.Tensor of shape same as
                self.content_feature

        Returns:
            Content cost as a scalar tensor
        """
        if (not isinstance(content_output, (tf.Tensor, tf.Variable)) or
                content_output.shape != self.content_feature.shape):
            raise TypeError(
                "content_output must be a tensor of shape {}".format(
                    self.content_feature.shape
                )
            )

        _, h, w, c = self.content_feature.shape

        diff = tf.square(content_output - self.content_feature)

        content_cost = tf.reduce_sum(diff) / tf.cast(
            h * w * c, tf.float32
        )

        return content_cost