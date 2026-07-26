#!/usr/bin/env python3
"""Module for creating a Ridge regression model."""

from sklearn import linear_model


def ridge_regression(random_state):
    """Return an untrained Ridge regression model."""
    model = linear_model.Ridge(random_state=random_state)
    return model
