import tensorflow as tf

def compute_grads(self, generated_image):
        """
        Calculates gradients for generated image

        Args:
            generated_image: tf.Tensor or tf.Variable of shape
                              (1, nh, nw, 3)

        Returns:
            gradients, J_total, J_content, J_style
        """

        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                generated_image.shape != self.content_image.shape):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(
                    self.content_image.shape
                )
            )

        with tf.GradientTape() as tape:
            J_total, J_content, J_style = self.total_cost(generated_image)

        gradients = tape.gradient(J_total, generated_image)

        return gradients, J_total, J_content, J_style