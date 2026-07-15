#!/usr/bin/env python3
"""Perform chi-square tests for categorical features."""

import pandas as pd
from scipy import stats


def chi_square_tests(df):
    """Return chi-square p-values for categorical features vs Churn."""
    p_values = {}

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for col in categorical_columns:
        if col != "Churn":
            contingency_table = pd.crosstab(df[col], df["Churn"])
            chi2, p_value, dof, expected = stats.chi2_contingency(
                contingency_table
            )
            p_values[col] = p_value

    return p_values
