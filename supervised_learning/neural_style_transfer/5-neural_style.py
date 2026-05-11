import tensorflow as tf

def style_cost(self, style_outputs):
        """
        Calculates the style cost for generated image

        Args:
            style_outputs: list of tf.Tensor outputs for generated image

        Returns:
            total style cost
        """

        if (not isinstance(style_outputs, list) or
                len(style_outputs) != len(self.style_layers)):
            raise TypeError(
                "style_outputs must be a list with a length of {}".format(
                    len(self.style_layers)
                )
            )

        L = len(self.style_layers)
        weight = 1 / L

        total_style_cost = 0

        for style_output, gram_target in zip(
            style_outputs, self.gram_style_features
        ):
            layer_cost = self.layer_style_cost(
                style_output,
                gram_target
            )
            total_style_cost += weight * layer_cost

        return total_style_cost