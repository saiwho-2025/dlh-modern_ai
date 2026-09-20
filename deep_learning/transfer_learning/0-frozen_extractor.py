#!/usr/bin/env python3
"""Build a frozen MobileNetV2 feature extractor."""
from tensorflow import keras as K


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

    # Add global average pooling to the frozen base model.
    inputs = keras.Input(shape=(224, 224, 3))
    features = base_model(inputs, training=False)
    outputs = keras.layers.GlobalAveragePooling2D()(features)

    # Return the feature extraction model.
    return keras.Model(inputs=inputs, outputs=outputs)
