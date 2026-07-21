#!/usr/bin/env python3
"""this function create a decision tree classifier using scikit-learn"""
from sklearn import tree


def build_decision_tree(
    min_samples_leaf: int,
    min_samples_split: int,
    random_state: int,
) -> tree.DecisionTreeClassifier:
    """
    Create a decision tree classifier using the Gini impurity criterion.

    The tree has no maximum depth, allowing it to grow until all leaves
    are pure or another stopping criterion is met.

    Args:
        min_samples_leaf: Minimum number of samples required at a leaf node.
        min_samples_split: Minimum number of samples required to split an
            internal node.
        random_state: Seed used for reproducible random number generation.

    Returns:
        A configured Scikit-learn DecisionTreeClassifier instance.
    """
    return tree.DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=random_state,
    )