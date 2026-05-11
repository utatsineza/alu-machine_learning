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