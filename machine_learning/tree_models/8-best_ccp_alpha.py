#!/usr/bin/env python3
"""Train and evaluate decision trees using cost-complexity pruning."""

from sklearn import tree

train_tree = __import__("1-train").train_tree


def prune_and_evaluate_trees(
    X_train,
    y_train,
    X_test,
    y_test,
    ccp_alphas,
    random_state,
    min_samples_leaf,
    min_samples_split,
):
    """
    Train and evaluate decision trees for multiple pruning alpha values.

    Args:
        X_train: Training feature matrix.
        y_train: Training target labels.
        X_test: Testing feature matrix.
        y_test: Testing target labels.
        ccp_alphas: Pruning alpha values used to train the classifiers.
        random_state: Integer seed used for reproducibility.
        min_samples_leaf: Minimum number of samples required at a leaf node.
        min_samples_split: Minimum number of samples required to split an
            internal node.

    Returns:
        A tuple containing:
            clfs: Trained DecisionTreeClassifier instances.
            train_scores: Training accuracy score for each classifier.
            test_scores: Testing accuracy score for each classifier.
    """
    clfs = []
    train_scores = []
    test_scores = []

    # Train one decision tree for each cost-complexity pruning value.
    for ccp_alpha in ccp_alphas:
        clf = tree.DecisionTreeClassifier(
            criterion="gini",
            max_depth=None,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            random
