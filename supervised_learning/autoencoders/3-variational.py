#!/usr/bin/env python3
"""
Creates a variational autoencoder (VAE)
"""

import tensorflow.keras as keras
from tensorflow.keras import layers, models, backend as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder

    Args:
        input_dims: integer, dimensions of the model input
        hidden_layers: list of integers, nodes for each hidden layer in encoder
        latent_dims: integer, dimensions of the latent space representation

    Returns:
        encoder: encoder model (outputs latent, mu, log_var)
        decoder: decoder model
        auto: full variational autoencoder model
    """

    # --- Encoder ---
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = layers.Dense(nodes, activation='relu')(x)

    # Latent mean and log variance
    mu = layers.Dense(latent_dims, activation=None, name='mu')(x)
    log_var = layers.Dense(latent_dims, activation=None, name='log_var')(x)

    # Sampling function using reparameterization trick
    def sampling(args):
        mu, log_var = args
        epsilon = K.random_normal(shape=K.shape(mu))
        return mu + K.exp(log_var / 2) * epsilon

    latent = layers.Lambda(sampling, output_shape=(latent_dims,), name='latent')([mu, log_var])

    encoder = models.Model(inputs, [latent, mu, log_var], name='encoder')

    # --- Decoder ---
    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input
    for nodes in reversed(hidden_layers):
        x = layers.Dense(nodes, activation='relu')(x)
    outputs = layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = models.Model(decoder_input, outputs, name='decoder')

    # --- Full VAE ---
    latent_sample = encoder(inputs)[0]
    reconstruction = decoder(latent_sample)
    auto = models.Model(inputs, reconstruction, name='vae')
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
