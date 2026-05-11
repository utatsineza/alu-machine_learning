import tensorflow as tf    
def content_cost(self, content_output):
        """
        Calculates the content cost for the generated image

        Args:
            content_output: tf.Tensor of shape same as self.content_feature

        Returns:
            content cost
        """

        if (not isinstance(content_output, (tf.Tensor, tf.Variable)) or
                content_output.shape != self.content_feature.shape):
            raise TypeError(
                "content_output must be a tensor of shape {}".format(
                    self.content_feature.shape
                )
            )

        # number of elements in feature map
        _, h, w, c = self.content_feature.shape

        # squared difference
        diff = tf.square(content_output - self.content_feature)

        # normalize
        content_cost = tf.reduce_sum(diff) / tf.cast(h * w * c, tf.float32)

        return content_cost