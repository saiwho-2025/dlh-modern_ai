#!/usr/bin/env python3
"""this function trains a decision tree classifier using scikit-learn"""
from sklearn import tree



def train_tree(clf, X, y) -> None:
    """
    Train a Scikit-learn tree-based classifier.

    Args:
        clf: A Scikit-learn classifier instance.
        X: Input features used to train the classifier.
        y: Target labels corresponding to the input features.

    Returns:
        None.
    """
    clf.fit(X, y)