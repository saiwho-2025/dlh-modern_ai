#!/usr/bin/env python3
"""Train and evaluate decision trees using cost-complexity pruning."""

from sklearn import tree

train_tree = __import__('1-train').train_tree


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
    Train and evaluate decision trees for multiple pruning values.

    Args:
        X_train: Training input features.
        y_train: Training target labels.
        X_test: Testing input features.
        y_test: Testing target labels.
        ccp_alphas: Cost-complexity pruning alpha values.
        random_state: Integer seed used for reproducibility.
        min_samples_leaf: Minimum samples required at a leaf node.
        min_samples_split: Minimum samples required to split an internal node.

    Returns:
        A tuple containing:
            clfs: A list of trained decision tree classifiers.
            train_scores: A list of training accuracy scores.
            test_scores: A list of testing accuracy scores.
    """
    clfs = []

    # Create and train one classifier for each pruning alpha value.
    for ccp_alpha in ccp_alphas:
        clf = tree.DecisionTreeClassifier(
            criterion='gini',
            max_depth=None,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            random_state=random_state,
            ccp_alpha=ccp_alpha,
        )
        train_tree(clf, X_train, y_train)
        clfs.append(clf)

    # Calculate training and testing accuracy for every trained classifier.
    train_scores = [
        clf.score(X_train, y_train)
        for clf in clfs
    ]
    test_scores = [
        clf.score(X_test, y_test)
        for clf in clfs
    ]

    return clfs, train_scores, test_scores
