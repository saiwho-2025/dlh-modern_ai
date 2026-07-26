#!/usr/bin/env python3

from sklearn import linear_model


def Linear_Regression():
    """Return an untrained ordinary least squares linear regression model."""
    model = linear_model.LinearRegression()
    return model
