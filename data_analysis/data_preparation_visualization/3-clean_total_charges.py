#!/usr/bin/env python3
"""a function handles missing values"""
import pandas as pd

def clean_total_charges(df, method='drop'):
    """the method handles different missing values in PandsDataFrame"""
    def to_numeric(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return float("nan")

    df["TotalCharges"] = df["TotalCharges"].apply(to_numeric)

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

    else:
        raise ValueError("method must be 'drop', 'median', or 'impute'")

    return df