"""
Variational Autoencoder (VAE) for anomaly detection.
Uses reparameterization trick and KL divergence loss.
Compatible with Keras 3 (TensorFlow 2.18+).
"""

import keras
from keras import layers, ops
from anomaly_detection.config import LATENT_DIM


class Sampling(layers.Layer):
    """Reparameterization trick: z = mu + sigma * epsilon"""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = keras.random.normal(shape=ops.shape(z_mean))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


class VAELossLayer(layers.Layer):
    """Custom layer that computes and adds VAE loss."""

    def call(self, inputs):
        original, reconstructed, z_mean, z_log_var = inputs
        reconstruction_loss = ops.mean(ops.square(original - reconstructed), axis=1)
        kl_loss = -0.5 * ops.mean(
            1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=1
        )
        total_loss = reconstruction_loss + 0.001 * kl_loss
        self.add_loss(ops.mean(total_loss))
        return reconstructed


def build_vae(input_dim: int, latent_dim: int = LATENT_DIM) -> tuple:
    """
    Encoder: input_dim -> 64 -> 32 -> (z_mean, z_log_var) -> sample
    Decoder: latent_dim -> 32 -> 64 -> input_dim
    Returns: (vae_model, encoder_model, decoder_model)
    """
    # Encoder
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(32, activation="relu")(latent_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(input_dim, activation="linear")(x)
    decoder = keras.Model(latent_inputs, outputs, name="decoder")

    # VAE: connect encoder -> decoder -> loss layer
    z_mean_out, z_log_var_out, z_out = encoder(inputs)
    reconstructed = decoder(z_out)
    vae_output = VAELossLayer()([inputs, reconstructed, z_mean_out, z_log_var_out])
    vae = keras.Model(inputs, vae_output, name="vae")
    vae.compile(optimizer="adam")

    return vae, encoder, decoder
