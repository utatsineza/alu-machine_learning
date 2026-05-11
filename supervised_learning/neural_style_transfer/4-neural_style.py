import tensorflow as tf

def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer

        Args:
            style_output: tensor of shape (1, h, w, c)
            gram_target: tensor of shape (1, c, c)

        Returns:
            Style cost for the layer
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