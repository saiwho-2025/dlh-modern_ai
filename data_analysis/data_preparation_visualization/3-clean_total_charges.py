#!/usr/bin/env python3
"""Clean missing values in TotalCharges."""


def clean_total_charges(df, method='drop'):
    """Handle missing values in TotalCharges."""
    cleaned_df = df.copy()

    if method == 'drop':
        cleaned_df = cleaned_df.dropna(subset=['TotalCharges'])

    elif method == 'median':
        cleaned_df['TotalCharges'] = cleaned_df['TotalCharges'].fillna(
            cleaned_df['TotalCharges'].median()
        )

    elif method == 'impute':
        cleaned_df['TotalCharges'] = cleaned_df['TotalCharges'].fillna(
            cleaned_df['MonthlyCharges'] * cleaned_df['tenure']
        )

    return cleaned_df
