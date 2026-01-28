#!/usr/bin/env python3
"""3-variational.py: Variational autoencoder using Keras"""

import tensorflow.keras as keras
from tensorflow.keras import backend as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder (VAE).

    Args:
        input_dims (int): Dimensions of the model input.
        hidden_layers (list): List of nodes for each hidden layer in the encoder.
        latent_dims (int): Dimensions of the latent space representation.

    Returns:
        encoder: Encoder model (outputs latent, mean, log variance).
        decoder: Decoder model.
        auto: Full VAE model.
    """
    # Sampling function
    def sampling(args):
        """Reparameterization trick to sample from N(mu, sigma^2)"""
        mu, log_sigma = args
        epsilon = K.random_normal(shape=K.shape(mu))
        return mu + K.exp(log_sigma / 2) * epsilon

    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    # Mean and log variance for latent space
    mu = keras.layers.Dense(latent_dims, activation=None, name='mu')(x)
    log_sigma = keras.layers.Dense(latent_dims, activation=None, name='log_sigma')(x)

    # Latent vector
    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,), name='latent')([mu, log_sigma])

    encoder = keras.Model(inputs, [z, mu, log_sigma], name='encoder')

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    # VAE: encoder -> decoder
    vae_outputs = decoder(encoder(inputs)[0])
    auto = keras.Model(inputs, vae_outputs, name='variational_autoencoder')

    # Compile
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
