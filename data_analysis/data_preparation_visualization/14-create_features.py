#!/usr/bin/env python3
"""Create engineered features from the dataset."""

import pandas as pd


def create_features(df):
    """Engineer NumServices and TenureGroup features."""
    engineered_df = df.copy()

    service_columns = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    engineered_df["NumServices"] = 0

    for col in service_columns:
        if col == "InternetService":
            engineered_df["NumServices"] += engineered_df[col].isin(
                ["DSL", "Fiber optic"]
            ).astype(int)
        else:
            engineered_df["NumServices"] += (
                engineered_df[col] == "Yes"
            ).astype(int)

    engineered_df["TenureGroup"] = pd.cut(
        engineered_df["tenure"],
        bins=[0, 12, 24, 48, 60, float("inf")],
        labels=["0-12", "13-24", "25-48", "49-60", "60+"]
    )

    engineered_df = engineered_df.drop(
        columns=service_columns + ["tenure"]
    )

    return engineered_df
