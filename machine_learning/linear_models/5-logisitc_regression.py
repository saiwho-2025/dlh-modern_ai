#!/usr/bin/env python3
"""Module for creating a logistic regression model."""

from sklearn import linear_model


def Logistic_Regression_Model(random_state):
    """Return an untrained LogisticRegression model."""
    model = linear_model.LogisticRegression(random_state=random_state)
    return model
