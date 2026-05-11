#!/usr/bin/env python3
"""Neural Style Transfer - Full implementation"""

import tensorflow as tf
import numpy as np


class NST:
    """Neural Style Transfer class"""

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]

    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image,
                 alpha=1e4, beta=1, var=10):
        """Constructor"""

        if (not isinstance(style_image, np.ndarray) or
                style_image.shape != style_image.shape or
                len(style_image.shape) != 3 or
                style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (not isinstance(content_image, np.ndarray) or
                len(content_image.shape) != 3 or
                content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (not isinstance(alpha, (int, float)) or alpha < 0):
            raise TypeError("alpha must be a non-negative number")

        if (not isinstance(beta, (int, float)) or beta < 0):
            raise TypeError("beta must be a non-negative number")

        if (not isinstance(var, (int, float)) or var < 0):
            raise TypeError("var must be a non-negative number")

        tf.enable_eager_execution()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)

        self.alpha = alpha
        self.beta = beta
        self.var = var

        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """Rescales image"""

        if (not isinstance(image, np.ndarray) or
                len(image.shape) != 3 or
                image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape

        max_dim = 512

        if h > w:
            new_h = max_dim
            new_w = int(w * max_dim / h)
        else:
            new_w = max_dim
            new_h = int(h * max_dim / w)

        img = tf.image.resize(
            image,
            (new_h, new_w),
            method=tf.image.ResizeMethod.BICUBIC
        )

        img = img / 255.0
        img = tf.clip_by_value(img, 0.0, 1.0)

        return tf.expand_dims(img, axis=0)

    def load_model(self):
        """Load VGG19 model"""

        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )
        vgg.trainable = False

        outputs = [vgg.get_layer(l).output for l in self.style_layers]
        outputs.append(vgg.get_layer(self.content_layer).output)

        self.model = tf.keras.Model(inputs=vgg.input, outputs=outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """Gram matrix"""

        if (not isinstance(input_layer, (tf.Tensor, tf.Variable)) or
                len(input_layer.shape) != 4):
            raise TypeError(
                "input_layer must be a tensor of rank 4"
            )

        _, h, w, c = input_layer.shape

        gram = tf.linalg.einsum('bijc,bijd->bcd',
                                input_layer,
                                input_layer)
        return gram / tf.cast(h * w, tf.float32)

    def generate_features(self):
        """Extract style and content features"""

        style = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255
        )
        content = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255
        )

        outputs_style = self.model(style)
        outputs_content = self.model(content)

        self.gram_style_features = [
            self.gram_matrix(out) for out in outputs_style[:-1]
        ]

        self.content_feature = outputs_content[-1]

    def layer_style_cost(self, style_output, gram_target):
        """Style cost per layer"""

        if (not isinstance(style_output, (tf.Tensor, tf.Variable)) or
                len(style_output.shape) != 4):
            raise TypeError(
                "style_output must be a tensor of rank 4"
            )

        _, _, _, c = style_output.shape

        if gram_target.shape != (1, c, c):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]".format(
                    c, c
                )
            )

        gram = self.gram_matrix(style_output)

        return tf.reduce_sum(tf.square(gram - gram_target)) / (c ** 2)

    def style_cost(self, style_outputs):
        """Total style cost"""

        if (not isinstance(style_outputs, list) or
                len(style_outputs) != len(self.style_layers)):
            raise TypeError(
                "style_outputs must be a list with a length of {}".format(
                    len(self.style_layers)
                )
            )

        total = 0
        weight = 1 / len(self.style_layers)

        for out, gram in zip(style_outputs, self.gram_style_features):
            total += weight * self.layer_style_cost(out, gram)

        return total

    def content_cost(self, content_output):
        """Content cost"""

        if (not isinstance(content_output, (tf.Tensor, tf.Variable)) or
                content_output.shape != self.content_feature.shape):
            raise TypeError(
                "content_output must be a tensor of shape {}".format(
                    self.content_feature.shape
                )
            )

        _, h, w, c = self.content_feature.shape

        return tf.reduce_sum(
            tf.square(content_output - self.content_feature)
        ) / (h * w * c)

    @staticmethod
    def variational_cost(generated_image):
        """Total variation loss"""

        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                len(generated_image.shape) != 4):
            raise TypeError(
                "generated_image must be a tensor of rank 4"
            )

        return tf.reduce_sum(
            tf.square(generated_image[:, :-1, :, :] -
                      generated_image[:, 1:, :, :]) +
            tf.square(generated_image[:, :, :-1, :] -
                      generated_image[:, :, 1:, :])
        )

    def total_cost(self, generated_image):
        """Total loss"""

        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                generated_image.shape != self.content_image.shape):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(
                    self.content_image.shape
                )
            )

        vgg = tf.keras.applications.vgg19
        prep = vgg.preprocess_input(generated_image * 255)

        outputs = self.model(prep)

        style = outputs[:-1]
        content = outputs[-1]

        Jc = self.content_cost(content)
        Js = self.style_cost(style)
        Jv = self.variational_cost(generated_image)

        J = self.alpha * Jc + self.beta * Js + self.var * Jv

        return J, Jc, Js, Jv

    def compute_grads(self, generated_image):
        """Compute gradients"""

        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                generated_image.shape != self.content_image.shape):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(
                    self.content_image.shape
                )
            )

        with tf.GradientTape() as tape:
            J, Jc, Js, Jv = self.total_cost(generated_image)

        grads = tape.gradient(J, generated_image)

        return grads, J, Jc, Js, Jv

    def generate_image(self, iterations=1000, step=None,
                       lr=0.01, beta1=0.9, beta2=0.99):
        """Train image"""

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")

        if step is not None:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError(
                    "iterations must be positive and less than iterations"
                )

        if not isinstance(lr, (int, float)):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")

        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if not 0 <= beta1 <= 1:
            raise ValueError("beta1 must be in the range [0, 1]")

        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if not 0 <= beta2 <= 1:
            raise ValueError("beta2 must be in the range [0, 1]")

        image = tf.Variable(self.content_image)

        optimizer = tf.train.AdamOptimizer(lr, beta1, beta2)

        best = float("inf")
        best_img = None

        for i in range(iterations + 1):

            grads, J, Jc, Js, Jv = self.compute_grads(image)
            optimizer.apply_gradients([(grads, image)])

            if J < best:
                best = J
                best_img = tf.identity(image)

            if step and (i % step == 0 or i == iterations):
                print(
                    "Cost at iteration {}: {}, content {}, style {}, var {}".format(
                        i, J, Jc, Js, Jv
                    )
                )

        return best_img, best