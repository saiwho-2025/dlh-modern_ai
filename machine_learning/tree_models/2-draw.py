#!/usr/bin/env python3
"""Display the textual structure of a trained decision tree classifier."""

from sklearn import tree


def draw(clf, feature_names, class_names) -> None:
    """
    Print a readable representation of a trained decision tree.

    Args:
        clf: A trained Scikit-learn DecisionTreeClassifier instance.
        feature_names: A list containing the input feature names.
        class_names: A list containing the target class names.

    Returns:
        None.
    """
    # Convert the trained decision tree into a readable text format.
    tree_structure = tree.export_text(
        clf,
        feature_names=feature_names,
        class_names=class_names,
    )

    # Display the decision rules and predicted classes.
    print(tree_structure)
