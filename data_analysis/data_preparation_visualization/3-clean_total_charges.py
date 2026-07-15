#!/usr/bin/env python3
"""a function handles missing values"""


def clean_total_charges(df, method='drop'):
    def to_float(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return float("nan")

    df["TotalCharges"] = df["TotalCharges"].apply(to_float)

    if method == "drop":
        df = df.dropna(subset=["TotalCharges"])

    elif method == "median":
        median = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(median)

    elif method == "impute":
        missing = df["TotalCharges"].isna()
        df.loc[missing, "TotalCharges"] = (
            df.loc[missing, "MonthlyCharges"] * df.loc[missing, "tenure"]
        )

    return df