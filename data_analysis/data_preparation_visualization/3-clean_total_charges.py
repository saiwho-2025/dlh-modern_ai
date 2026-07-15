#!/usr/bin/env python3
"""a function handles missing values"""


def clean_total_charges(df, method='drop'):
    """the method handles different missing values in PandsDataFrame"""
    def convert(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return float("nan")

    df["TotalCharges"] = df["TotalCharges"].apply(convert)

    if method == "drop":
        df = df.dropna(subset=["TotalCharges"])

    elif method == "median":
        df["TotalCharges"] = df["TotalCharges"].fillna(
            df["TotalCharges"].median()
        )

    elif method == "impute":
        df["TotalCharges"] = df["TotalCharges"].fillna(
            df["MonthlyCharges"] * df["tenure"]
        )

    return df