#!/usr/bin/env python3
"""the function removes customerID column """


def drop_costomerID(df):
    """the function removes the column and returns updated dp"""
    cleaned_df = df.copy()
    cleaned_df = df.drop(columns = ["customerID"])
    return cleaned_df
