#!/usr/bin/env python3
"""Retrieve the cost-complexity pruning path of a decision tree."""


def get_pruning_path(clf, X, y):
    """
    Retrieve the cost-complexity pruning path for a decision tree.

    Args:
        clf: A Scikit-learn DecisionTreeClassifier instance.
        X: Input features used to calculate the pruning path.
        y: Target labels corresponding to the input features.

    Returns:
        A tuple containing:
            ccp_alphas: Effective alpha values used for pruning.
            impurities: Total leaf impurity for each alpha value.
    """
    # Calculate the effective alpha values and corresponding impurities.
    pruning_path = clf.cost_complexity_pruning_path(X, y)

    # Extract the pruning values from the returned result.
    ccp_alphas = pruning_path.ccp_alphas
    impurities = pruning_path.impurities

    return ccp_alphas, impurities
