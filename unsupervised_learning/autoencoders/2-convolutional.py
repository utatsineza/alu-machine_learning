#!/usr/bin/env python3
"""
Creates a convolutional autoencoder
"""

import tensorflow.keras as keras
from tensorflow.keras import layers, models


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder

    Args:
        input_dims: tuple, dimensions of the model input (H, W, C)
        filters: list of integers, number of filters for each conv layer in encoder
        latent_dims: tuple, dimensions of the latent space representation

    Returns:
        encoder: encoder model
        decoder: decoder model
        auto: full autoencoder model
    """
    # --- Encoder ---
    input_layer = keras.Input(shape=input_dims)
    x = input_layer

    for f in filters:
        x = layers.Conv2D(f, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Latent representation
    latent = layers.Conv2D(latent_dims[2], kernel_size=(3, 3),
                           padding='same', activation='relu')(x)
    encoder = models.Model(inputs=input_layer, outputs=latent, name='encoder')

    # --- Decoder ---
    decoder_input = keras.Input(shape=latent_dims)
    x = decoder_input

    reversed_filters = list(reversed(filters))
    for i, f in enumerate(reversed_filters):
        # Last two convs have special handling
        if i == len(reversed_filters) - 1:
            # Second to last conv: valid padding, relu
            x = layers.Conv2D(f, kernel_size=(3, 3), padding='valid', activation='relu')(x)
        else:
            x = layers.Conv2D(f, kernel_size=(3, 3), padding='same', activation='relu')(x)
            x = layers.UpSampling2D((2, 2))(x)

    # Last conv: same number of channels as input, sigmoid, no upsampling
    output_layer = layers.Conv2D(input_dims[2], kernel_size=(3, 3),
                                 padding='same', activation='sigmoid')(x)

    decoder = models.Model(inputs=decoder_input, outputs=output_layer, name='decoder')

    # --- Full Autoencoder ---
    auto_input = input_layer
    encoded = encoder(auto_input)
    reconstructed = decoder(encoded)
    auto = models.Model(inputs=auto_input, outputs=reconstructed, name='conv_autoencoder')
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
