#!/usr/bin/env python3
"""Select the best cost-complexity pruning alpha and classifier."""


def get_best_alpha(clfs, train_scores, test_scores, ccp_alphas):
    """
    Select the best pruning alpha and its corresponding classifier.

    Models are ranked by highest test accuracy, smallest difference
    between training and test accuracy, and largest pruning alpha.

    Args:
        clfs: List of trained DecisionTreeClassifier instances.
        train_scores: Training accuracy score for each classifier.
        test_scores: Testing accuracy score for each classifier.
        ccp_alphas: Pruning alpha value associated with each classifier.

    Returns:
        A tuple containing:
            best_alpha: The selected cost-complexity pruning alpha.
            best_clf: The classifier associated with the selected alpha.
    """
    # Rank models by test accuracy, generalization gap, and pruning strength.
    best_index = max(
        range(len(clfs)),
        key=lambda index: (
            test_scores[index],
            -abs(train_scores[index] - test_scores[index]),
            ccp_alphas[index],
        ),
    )

    # Return the pruning alpha and classifier at the best-ranked position.
    best_alpha = ccp_alphas[best_index]
    best_clf = clfs[best_index]

    return best_alpha, best_clf
