import tensorflow as tf

def total_cost(self, generated_image):
        """
        Calculates total cost for generated image

        Args:
            generated_image: tf.Tensor of shape same as self.content_image

        Returns:
            (J, J_content, J_style)
        """

        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                generated_image.shape != self.content_image.shape):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(
                    self.content_image.shape
                )
            )

        # preprocess generated image
        vgg19 = tf.keras.applications.vgg19
        preprocessed = vgg19.preprocess_input(generated_image * 255)

        outputs = self.model(preprocessed)

        # split outputs
        style_outputs = outputs[:-1]
        content_output = outputs[-1]

        # compute costs
        J_content = self.content_cost(content_output)
        J_style = self.style_cost(style_outputs)

        # total cost
        J = (self.alpha * J_content) + (self.beta * J_style)

        return J, J_content, J_style