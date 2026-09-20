#!/usr/bin/env python3
"""Add a classification head to a pretrained feature extractor."""

from tensorflow import keras as K


def add_classification_head(base_model, num_classes):
    """Return a model with a classification head."""
    # Get the pooled feature vector from the base model.
    features = base_model.output

    # Add a fully connected hidden layer.
    features = K.layers.Dense(
        128,
        activation="relu",
    )(features)

    # Add the final classification layer.
    outputs = K.layers.Dense(
        num_classes,
        activation="softmax",
    )(features)

    # Build and return the complete classification model.
    return K.Model(
        inputs=base_model.input,
        outputs=outputs,
    )
