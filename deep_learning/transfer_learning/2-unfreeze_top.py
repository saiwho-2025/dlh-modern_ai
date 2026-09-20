#!/usr/bin/env python3
"""Unfreeze the top layers of a pretrained base model."""


def unfreeze_top_layers(model, n_layers):
    """Unfreeze the last n layers of the base model."""
    # Get the base model, which follows the input layer.
    base_model = model.layers[1]

    # Allow individual layers of the base model to be trainable.
    base_model.trainable = True

    # Freeze all layers in the base model first.
    for layer in base_model.layers:
        layer.trainable = False

    # Unfreeze only the requested number of final layers.
    if n_layers > 0:
        for layer in base_model.layers[-n_layers:]:
            layer.trainable = True

    return None
