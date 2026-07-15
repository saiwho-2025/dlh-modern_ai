#!/usr/bin/env python3
"""the function removes duplicate rows"""


def remove_duplicates(df):
    """drop all duplicate rows and return the duplicated dataframe"""
    cleaned_df = df.drop_duplicates()
    return cleaned_df
