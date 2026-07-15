#!/usr/bin/env python3
#!/usr/bin/env python3
"""Clean missing values in TotalCharges."""


def clean_total_charges(df, method='drop'):
    """Handle missing values in the TotalCharges column."""
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