#!/usr/bin/env python3
"""Build a frozen MobileNetV2 feature extractor."""

import matplotlib.pyplot as plt
import seaborn as sns


def build_feature_extractor():
    """Return a frozen MobileNetV2 feature extractor."""
    # Load MobileNetV2 without its classification head.
    base_model = keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )

    # Freeze the pretrained convolutional base.
    base_model.trainable = False

    # Define the model input.
    inputs = keras.Input(shape=(224, 224, 3))

    # Extract convolutional features without updating frozen layers.
    features = base_model(inputs, training=False)

    # Convert feature maps into one feature vector per image.
    outputs = keras.layers.GlobalAveragePooling2D()(features)

    # Return the feature extraction model.
    return keras.Model(inputs=inputs, outputs=outputs)
