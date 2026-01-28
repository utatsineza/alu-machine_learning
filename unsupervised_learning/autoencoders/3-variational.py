#!/usr/bin/env python3
"""3-variational.py: Variational autoencoder using Keras"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Args:
        input_dims (int): Dimensions of the input.
        hidden_layers (list): Number of nodes for each hidden layer in
            the encoder.
        latent_dims (int): Dimensions of the latent space representation.

    Returns:
        encoder: Encoder model that outputs latent vector, mean, and log
            variance.
        decoder: Decoder model.
        auto: Full variational autoencoder model.
    """
    K = keras.backend  # Access backend through keras only

    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    z_mean = keras.layers.Dense(latent_dims, name='z_mean')(x)
    z_log_var = keras.layers.Dense(latent_dims, name='z_log_var')(x)

    # Sampling function
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean))
        return z_mean + K.exp(z_log_var / 2) * epsilon

    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,),
                            name='z')([z_mean, z_log_var])

    encoder = keras.Model(inputs, [z, z_mean, z_log_var], name='encoder')

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    # Full VAE
    auto_outputs = decoder(z)
    auto = keras.Model(inputs, auto_outputs, name='variational_autoencoder')

    # Loss: reconstruction + KL divergence
    reconstruction_loss = keras.losses.binary_crossentropy(inputs, auto_outputs)
    reconstruction_loss *= input_dims
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1) * -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    auto.add_loss(vae_loss)
    auto.compile(optimizer='adam')

    return encoder, decoder, auto
