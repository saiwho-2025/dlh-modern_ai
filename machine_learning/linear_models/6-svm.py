#!/usr/bin/env python3
"""Module for creating Support Vector Machine classifiers."""

from sklearn import svm


def get_SVM_model(name, random_state):
    """Return an untrained SVC model with the specified kernel."""
    model = svm.SVC(kernel=name, random_state=random_state)
    return model
