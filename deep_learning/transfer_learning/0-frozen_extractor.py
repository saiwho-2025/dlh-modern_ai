#!/usr/bin/env python3
"""Build a frozen MobileNetV2 feature extractor."""

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model


def build_feature_extractor():
    """Return a frozen MobileNetV2 model for image feature extraction."""
    # Load MobileNetV2 without its ImageNet classification head.
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )

    # Freeze all pretrained MobileNetV2 weights.
    base_model.trainable = False

    # Pass images through the frozen convolutional base.
    inputs = base_model.input
    features = base_model(inputs, training=False)

    # Convert feature maps into a single feature vector per image.
    outputs = GlobalAveragePooling2D()(features)

    # Return the complete feature extraction model.
    return Model(
        inputs=inputs,
        outputs=outputs,
        name="feature_extractor",
    )
