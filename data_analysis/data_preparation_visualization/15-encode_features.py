#!/usr/bin/env python3
"""Encode features for modeling."""

import pandas as pd
from sklearn import preprocessing


def encode_features(df):
    """Encode categorical features for modeling."""
    encoded_df = df.copy()

    churn_encoder = preprocessing.LabelEncoder()
    encoded_df["Churn"] = churn_encoder.fit_transform(encoded_df["Churn"])

    binary_columns = [
        "Partner",
        "Dependents",
        "PaperlessBilling",
        "SeniorCitizen"
    ]

    binary_encoder = preprocessing.OrdinalEncoder(
        categories=[["No", "Yes"]]
    )

    for col in binary_columns:
        encoded_df[[col]] = binary_encoder.fit_transform(encoded_df[[col]])
        encoded_df[col] = encoded_df[col].astype(int)

    tenure_encoder = preprocessing.OrdinalEncoder()
    encoded_df[["TenureGroup"]] = tenure_encoder.fit_transform(
        encoded_df[["TenureGroup"]]
    )
    encoded_df["TenureGroup"] = encoded_df["TenureGroup"].astype(int)

    encoded_df = pd.get_dummies(
        encoded_df,
        columns=["Contract", "PaymentMethod"],
        drop_first=True
    )

    return encoded_df, churn_encoder, binary_encoder, tenure_encoder
