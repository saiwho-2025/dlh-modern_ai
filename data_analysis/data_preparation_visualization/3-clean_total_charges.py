#!/usr/bin/env python3
"""a function handles missing values"""


def clean_total_charges(df,method='drop'):
    if method == "drop":
        df = df.dropna(subset=["TotalCharges"])

    elif method == "median":
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    elif method == "impute":
        df["TotalCharges"] = df["TotalCharges"].fillna(
            df["MonthlyCharges"] * df["tenure"]
        )

    else:
        raise ValueError("method must be one of: 'drop', 'median', or 'impute'")

    return df