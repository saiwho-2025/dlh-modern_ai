#!/usr/bin/env python3
"""Create boosting classifiers using supported machine-learning libraries."""

from sklearn import ensemble
import lightgbm as lgb
import xgboost as xgb


def compare_boosting_classifiers(name, n_estimators, random_state):
    """
    Initialize a boosting classifier based on its algorithm name.

    Args:
        name: Name of the boosting algorithm to initialize.
        n_estimators: Number of boosting iterations or trees.
        random_state: Random seed used to ensure reproducibility.

    Returns:
        An untrained instance of the selected boosting classifier.

    Raises:
        ValueError: If the specified model name is not supported.
    """
    # Initialize the classifier matching the requested algorithm.
    if name == "adaboost":
        model = ensemble.AdaBoostClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
        )
    elif name == "gradientboosting":
        model = ensemble.GradientBoostingClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
        )
    elif name == "xgboost":
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
        )
    elif name == "lightgbm":
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            verbose=-1,
        )
    else:
        raise ValueError(f"Unknown model name '{name}'")

    return model
