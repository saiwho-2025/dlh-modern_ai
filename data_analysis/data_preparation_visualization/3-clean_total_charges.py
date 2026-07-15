#!/usr/bin/env python3
"""a function handles missing values"""
import pandas as pd

def clean_total_charges(df, method='drop'):
    """the method handles different missing values in PandsDataFrame"""
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

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