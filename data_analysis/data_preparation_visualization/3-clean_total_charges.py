#!/usr/bin/env python3
"""a function handles missing values"""


def clean_total_charges(df, method='drop'):
    """try put drop in the last"""
    def to_float(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return float("nan")

    df["TotalCharges"] = df["TotalCharges"].apply(to_float)

    if method == "median":
        df["TotalCharges"] = df["TotalCharges"].fillna(
            df["TotalCharges"].median()
        )

    elif method == "impute":
        missing = df["TotalCharges"].isna()
        df.loc[missing, "TotalCharges"] = (
            df.loc[missing, "MonthlyCharges"] * df.loc[missing, "tenure"]
        )

    elif method == "drop":
        df = df.dropna(subset=["TotalCharges"])

    return df