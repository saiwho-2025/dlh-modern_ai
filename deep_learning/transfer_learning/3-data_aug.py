#!/usr/bin/env python3
"""Build an image data augmentation model."""

import tensorflow as tf


def build_data_augmentation():
    """Return a Sequential model for image data augmentation."""
    # Create the augmentation pipeline.
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(
            "horizontal",
            seed=42,
        ),
        tf.keras.layers.RandomRotation(
            0.15,
            seed=42,
        ),
        tf.keras.layers.RandomZoom(
            0.15,
            seed=42,
        ),
        tf.keras.layers.RandomContrast(
            0.1,
            seed=42,
        ),
    ])

    # Return the augmentation model.
    return data_augmentation
