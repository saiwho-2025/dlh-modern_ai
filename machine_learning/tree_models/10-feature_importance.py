#!/usr/bin/env python3
"""Compute feature importances from a trained random forest classifier."""

import numpy as np


def feature_importance(rf):
    """
    Compute and sort feature importance scores from a random forest.

    Args:
        rf: A trained Scikit-learn RandomForestClassifier instance.

    Returns:
        A tuple containing:
            importances: Feature importance scores.
            indices: Feature indices sorted in ascending order of importance.
    """
    # Retrieve the importance score assigned to each input feature.
    importances = rf.feature_importances_

    # Sort feature indices from the least to the most important.
    indices = np.argsort(importances)

    return importances, indices
