#!/usr/bin/env python3
"""Module that creates an ordinary least squares linear regression model."""

from sklearn import linear_model


def Linear_Regression():
    """Return an untrained LinearRegression model."""
    return linear_model.LinearRegression()
