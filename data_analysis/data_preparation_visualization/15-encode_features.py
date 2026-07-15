#!/usr/bin/env python3
"""Encode features for modeling."""

import pandas as pd
from sklearn import preprocessing


def encode_features(df):
    """Encode categorical features for modeling."""
    encoded_df = df.copy()

    # Churn: No -> 0, Yes -> 1
    churn_encoder = preprocessing.LabelEncoder()
    encoded_df["Churn"] = churn_encoder.fit_transform(encoded_df["Churn"])

    # Binary columns: No -> 0, Yes -> 1
    binary_columns = [
        "Partner",
        "Dependents",
        "PaperlessBilling",
        "SeniorCitizen"
    ]

    binary_encoder = preprocessing.OrdinalEncoder(
        categories=[["No", "Yes"]] * len(binary_columns)
    )
    encoded_df[binary_columns] = binary_encoder.fit_transform(
        encoded_df[binary_columns]
    )

    # TenureGroup: alphabetical order
    tenure_encoder = preprocessing.OrdinalEncoder()
    encoded_df[["TenureGroup"]] = tenure_encoder.fit_transform(
        encoded_df[["TenureGroup"]]
    )

    # One-hot encoding, drop first category
    encoded_df = pd.get_dummies(
        encoded_df,
        columns=["Contract", "PaymentMethod"],
        drop_first=True
    )

    return encoded_df, churn_encoder, binary_encoder, tenure_encoder
 