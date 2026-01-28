#!/usr/bin/env python3
"""2-convolutional.py: Convolutional autoencoder using Keras"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder.

    Args:
        input_dims (tuple): Dimensions of the model input (height, width, channels).
        filters (list): Number of filters for each convolutional layer in the encoder.
        latent_dims (tuple): Dimensions of the latent space representation (height, width, channels).

    Returns:
        encoder: Encoder model.
        decoder: Decoder model.
        auto: Full convolutional autoencoder model.
    """
    # Input layer
    input_layer = keras.Input(shape=input_dims)
    x = input_layer

    # Encoder: conv + relu + maxpool
    for f in filters:
        x = keras.layers.Conv2D(
            filters=f, kernel_size=(3, 3), padding='same', activation='relu'
        )(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Latent representation
    latent = keras.layers.Conv2D(
        filters=latent_dims[2],
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    )(x)

    encoder = keras.Model(inputs=input_layer, outputs=latent, name="encoder")

    # Decoder: conv + relu + upsample
    decoder_input = keras.Input(shape=latent_dims)
    x = decoder_input
    rev_filters = list(reversed(filters))
    for i, f in enumerate(rev_filters):
        if i < len(rev_filters) - 2:
            x = keras.layers.Conv2D(
                filters=f, kernel_size=(3, 3), padding='same', activation='relu'
            )(x)
            x = keras.layers.UpSampling2D(size=(2, 2))(x)
        elif i == len(rev_filters) - 2:
            x = keras.layers.Conv2D(
                filters=f, kernel_size=(3, 3), padding='valid', activation='relu'
            )(x)
            x = keras.layers.UpSampling2D(size=(2, 2))(x)
        else:
            # Last conv layer: same filters as input channels, sigmoid activation
            x = keras.layers.Conv2D(
                filters=input_dims[2], kernel_size=(3, 3), padding='same', activation='sigmoid'
            )(x)

    decoder = keras.Model(inputs=decoder_input, outputs=x, name="decoder")

    # Full autoencoder
    auto_output = decoder(encoder(input_layer))
    auto = keras.Model(inputs=input_layer, outputs=auto_output, name="conv_autoencoder")

    # Compile autoencoder
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
