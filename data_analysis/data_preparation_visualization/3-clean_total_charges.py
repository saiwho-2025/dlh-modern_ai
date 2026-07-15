#!/usr/bin/env python3
"""a function that cleans data of NA, set"""


def clean_total_charges(df, method='drop'):
    """ normalize and drop"""
    def to_float(value):
        """value type float"""
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