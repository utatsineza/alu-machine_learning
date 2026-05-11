import tensorflow as tf

def generate_image(self, iterations=1000, step=None,
                       lr=0.01, beta1=0.9, beta2=0.99):
        """
        Generates neural style transferred image

        Returns:
            (best_image, best_cost)
        """

        # -------------------- validations --------------------
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")

        if step is not None:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError(
                    "step must be positive and less than iterations"
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

        # -------------------- initialization --------------------
        generated_image = tf.Variable(self.content_image)

        optimizer = tf.train.AdamOptimizer(
            lr,
            beta1=beta1,
            beta2=beta2
        )

        best_cost = float("inf")
        best_image = None

        # -------------------- optimization loop --------------------
        for i in range(iterations + 1):

            grads, J_total, J_content, J_style = self.compute_grads(
                generated_image
            )

            optimizer.apply_gradients([(grads, generated_image)])

            # track best
            if J_total < best_cost:
                best_cost = J_total
                best_image = tf.identity(generated_image)

            # print progress
            if step is not None and (i % step == 0 or i == iterations):
                print(
                    "Cost at iteration {}: {}, content {}, style {}".format(
                        i, J_total, J_content, J_style
                    )
                )

        return best_image, best_cost