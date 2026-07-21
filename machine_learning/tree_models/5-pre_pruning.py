#!/usr/bin/env python3
"""Find the best pre-pruning parameters for a decision tree classifier."""

from sklearn import model_selection


def prepruning(X, y, clf):
    """
    Find the best decision tree pre-pruning hyperparameters.

    Args:
        X: Input features used to train the classifier.
        y: Target labels corresponding to the input features.
        clf: An untrained Scikit-learn DecisionTreeClassifier instance.

    Returns:
        A dictionary containing the best hyperparameter combination.
    """
    # Define the decision tree hyperparameters explored by the grid search.
    parameter_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": range(2, 5),
        "min_samples_leaf": range(2, 5),
        "min_samples_split": range(2, 5),
    }

    # Evaluate every hyperparameter combination using cross-validation.
    grid_search = model_selection.GridSearchCV(
        estimator=clf,
        param_grid=parameter_grid,
    )
    grid_search.fit(X, y)

    # Return the parameters associated with the best-performing model.
    return grid_search.best_params_
