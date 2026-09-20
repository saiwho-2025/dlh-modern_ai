#!/usr/bin/env python3
"""Unfreeze the top layers of a pretrained model."""


def unfreeze_top_layers(model, n_layers):
    """Unfreeze the last n layers of the model."""
    # Freeze all layers in the base model.
    for layer in model.layers:
        layer.trainable = False

    # Unfreeze only the requested last layers.
    if n_layers > 0:
        for layer in model.layers[-n_layers:]:
            layer.trainable = True

    return None
