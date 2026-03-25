"""
1D-CNN Autoencoder for anomaly detection.
Treats each feature as a position in a 1D sequence.
"""

import tensorflow as tf
from anomaly_detection.config import ENCODING_DIM


def build_cnn_autoencoder(input_dim: int, encoding_dim: int = ENCODING_DIM) -> tf.keras.Model:
    """
    Input: (batch, input_dim, 1) -- each feature as a 1D channel.
    Encoder: Conv1D(32,3) -> Conv1D(16,3) -> GlobalAvgPool -> Dense(encoding_dim)
    Decoder: Dense -> Reshape -> Conv1DTranspose -> Conv1DTranspose -> Conv1D(1)
    """
    inputs = tf.keras.Input(shape=(input_dim, 1))

    # Encoder
    x = tf.keras.layers.Conv1D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(16, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    encoded = tf.keras.layers.Dense(encoding_dim, activation="relu", name="bottleneck")(x)

    # Decoder
    x = tf.keras.layers.Dense(input_dim * 16, activation="relu")(encoded)
    x = tf.keras.layers.Reshape((input_dim, 16))(x)
    x = tf.keras.layers.Conv1DTranspose(16, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1DTranspose(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    decoded = tf.keras.layers.Conv1D(1, 3, padding="same", activation="linear")(x)

    model = tf.keras.Model(inputs, decoded, name="cnn_autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model
