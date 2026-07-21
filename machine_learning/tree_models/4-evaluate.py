#!/usr/bin/env python3
"""Evaluate classifier predictions using a classification report."""

from sklearn import metrics


def evaluate(true_labels, predicted_labels, class_names):
    """
    Generate a detailed classification report.

    Args:
        true_labels: Ground truth labels.
        predicted_labels: Labels predicted by the classifier.
        class_names: Class names corresponding to the label indices.

    Returns:
        A string containing precision, recall, F1-score, and support
        for each class.
    """
    # Include every supplied class, even when it has no predicted samples.
    labels = range(len(class_names))

    # Generate and return the classification report as formatted text.
    report = metrics.classification_report(
        true_labels,
        predicted_labels,
        labels=labels,
        target_names=class_names,
        zero_division=0,
    )

    return report
