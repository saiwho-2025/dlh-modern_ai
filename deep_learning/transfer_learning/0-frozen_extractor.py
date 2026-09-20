#!/usr/bin/env python3
"""Build a frozen MobileNetV2 feature extractor."""

from tensorflow import keras as K


def build_feature_extractor():
    """Return a frozen MobileNetV2 feature extractor."""
    # Load MobileNetV2 without its classification head.
    base_model = K.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )

    # Freeze the pretrained base model.
    base_model.trainable = False

    # Build the feature extractor.
    inputs = K.Input(shape=(224, 224, 3))
    features = base_model(inputs)
    outputs = K.layers.GlobalAveragePooling2D()(features)

    # Return the complete model.
    return K.Model(inputs=inputs, outputs=outputs)
