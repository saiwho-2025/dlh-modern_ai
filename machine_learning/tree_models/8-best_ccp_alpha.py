#!/usr/bin/env python3
"""Create a random forest classifier."""


def random_forest(n_estimators, random_state):
    """
    Create a Scikit-learn random forest classifier.

    Args:
        n_estimators: Number of decision trees in the forest.
        random_state: Seed used for reproducible random number generation.

    Returns:
        A configured Scikit-learn RandomForestClassifier instance.
    """
    # Configure the forest with the requested number of decision trees.
    model = ensemble.RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
    )

    return model
