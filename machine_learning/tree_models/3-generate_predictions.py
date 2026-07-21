#!/usr/bin/env python3
"""Generate predictions using a trained tree-based classifier."""


def generate_predictions(clf, X):
    """
    Generate class predictions for the provided input samples.

    Args:
        clf: A trained Scikit-learn classifier instance.
        X: Feature matrix provided as a NumPy array or pandas DataFrame.

    Returns:
        A NumPy array containing the predicted class labels.
    """
    # Use the trained classifier to predict one class for each input sample.
    predictions = clf.predict(X)

    return predictions
