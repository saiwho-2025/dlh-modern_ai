#!/usr/bin/env python3
"""the function removes customerID column """


def drop_customerID(df):
    """Remove the customerID column."""
    cleaned_df = df.copy()
    cleaned_df = cleaned_df.drop(columns=["customerID"])
    return cleaned_df
