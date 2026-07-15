#!/usr/bin/env python3
"""Perform Welch's t-tests for numeric features."""

from scipy import stats


def ttest_numeric(df):
    """Return Welch's t-test p-values for numeric features vs Churn."""
    p_values = {}

    numeric_columns = df.select_dtypes(include="number").columns

    for col in numeric_columns:
        churn_yes = df[df["Churn"] == "Yes"][col].dropna()
        churn_no = df[df["Churn"] == "No"][col].dropna()

        t_stat, p_value = stats.ttest_ind(
            churn_yes,
            churn_no,
            equal_var=False
        )

        p_values[col] = p_value

    return p_values
