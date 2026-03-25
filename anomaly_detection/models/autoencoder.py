"""
Vanilla Dense Autoencoder for anomaly detection.
Symmetric encoder-decoder architecture with a bottleneck.
"""

import tensorflow as tf
from anomaly_detection.config import ENCODING_DIM


def build_autoencoder(input_dim: int, encoding_dim: int = ENCODING_DIM) -> tf.keras.Model:
    """
    Encoder: input_dim -> 64 -> 32 -> 16 -> encoding_dim
    Decoder: encoding_dim -> 16 -> 32 -> 64 -> input_dim
    """
    inputs = tf.keras.Input(shape=(input_dim,))

    # Encoder
    x = tf.keras.layers.Dense(64, activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    encoded = tf.keras.layers.Dense(encoding_dim, activation="relu", name="bottleneck")(x)

    # Decoder
    x = tf.keras.layers.Dense(16, activation="relu")(encoded)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    decoded = tf.keras.layers.Dense(input_dim, activation="linear")(x)

    model = tf.keras.Model(inputs, decoded, name="autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model
