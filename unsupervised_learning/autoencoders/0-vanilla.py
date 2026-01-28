#!/usr/bin/env python3
"""
Creates a vanilla autoencoder
"""

import tensorflow.keras as keras
from tensorflow.keras import layers, models, optimizers


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates an autoencoder

    Args:
        input_dims: integer, dimensions of the model input
        hidden_layers: list of integers, nodes for each hidden layer in encoder
        latent_dims: integer, dimensions of the latent space representation

    Returns:
        encoder: the encoder model
        decoder: the decoder model
        auto: the full autoencoder model
    """
    # --- Encoder ---
    input_layer = keras.Input(shape=(input_dims,))
    x = input_layer

    for nodes in hidden_layers:
        x = layers.Dense(nodes, activation='relu')(x)

    latent = layers.Dense(latent_dims, activation='relu')(x)
    encoder = models.Model(inputs=input_layer, outputs=latent, name='encoder')

    # --- Decoder ---
    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input

    for nodes in reversed(hidden_layers):
        x = layers.Dense(nodes, activation='relu')(x)

    output_layer = layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = models.Model(inputs=decoder_input, outputs=output_layer, name='decoder')

    # --- Autoencoder ---
    auto_input = input_layer
    encoded = encoder(auto_input)
    reconstructed = decoder(encoded)
    auto = models.Model(inputs=auto_input, outputs=reconstructed, name='autoencoder')
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
