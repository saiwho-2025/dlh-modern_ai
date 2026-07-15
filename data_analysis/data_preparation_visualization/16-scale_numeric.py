#!/usr/bin/env python3
"""Standardize numeric columns."""

from sklearn import preprocessing


def scale_numeric(df):
    """Scale MonthlyCharges and TotalCharges using StandardScaler."""
    scaled_df = df.copy()

    scaler = preprocessing.StandardScaler()

    columns_to_scale = ["MonthlyCharges", "TotalCharges"]

    scaled_df[columns_to_scale] = scaler.fit_transform(
        scaled_df[columns_to_scale]
    )

    return scaled_df
