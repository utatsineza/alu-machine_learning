#!/usr/bin/env python3
"""0-vanilla.py: Vanilla autoencoder using Keras"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a vanilla autoencoder.

    Args:
        input_dims (int): Dimensions of the input.
        hidden_layers (list): List of nodes for each hidden layer in
            the encoder.
        latent_dims (int): Dimensions of the latent space
            representation.

    Returns:
        encoder: Encoder model.
        decoder: Decoder model.
        auto: Full autoencoder model.
    """
    # Input layer
    input_layer = keras.Input(shape=(input_dims,))

    # Encoder
    x = input_layer
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    latent = keras.layers.Dense(latent_dims, activation='relu')(x)

    encoder = keras.Model(inputs=input_layer, outputs=latent, name="encoder")

    # Decoder
    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    output_layer = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(inputs=decoder_input, outputs=output_layer,
                          name="decoder")

    # Full autoencoder
    auto_output = decoder(encoder(input_layer))
    auto = keras.Model(inputs=input_layer, outputs=auto_output,
                       name="autoencoder")

    # Compile
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
