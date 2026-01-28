#!/usr/bin/env python3
"""2-convolutional.py: Convolutional autoencoder using Keras"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder.

    Args:
        input_dims (tuple): Dimensions of the input (H, W, C).
        filters (list): Number of filters for each conv layer in the encoder.
            The decoder reverses this list.
        latent_dims (tuple): Dimensions of the latent space representation
            (H, W, C).

    Returns:
        encoder: Encoder model.
        decoder: Decoder model.
        auto: Full convolutional autoencoder model.
    """
    # Encoder
    inputs = keras.Input(shape=input_dims)
    x = inputs
    for i, f in enumerate(filters):
        x = keras.layers.Conv2D(filters=f, kernel_size=(3, 3),
                                padding='same', activation='relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    latent = keras.layers.Conv2D(filters=latent_dims[2], kernel_size=(3, 3),
                                 padding='same', activation='relu')(x)

    encoder = keras.Model(inputs, latent, name='encoder')

    # Decoder
    latent_inputs = keras.Input(shape=latent_dims)
    x = latent_inputs
    for i, f in enumerate(reversed(filters[:-1])):
        x = keras.layers.Conv2D(filters=f, kernel_size=(3, 3),
                                padding='same', activation='relu')(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)

    # Second to last conv layer with valid padding
    x = keras.layers.Conv2D(filters=filters[0], kernel_size=(3, 3),
                            padding='valid', activation='relu')(x)
    # Output layer: same channels as input
    outputs = keras.layers.Conv2D(filters=input_dims[2], kernel_size=(3, 3),
                                  padding='same', activation='sigmoid')(x)

    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    # Full autoencoder
    auto_outputs = decoder(encoder(inputs))
    auto = keras.Model(inputs, auto_outputs, name='conv_autoencoder')

    # Compile
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
