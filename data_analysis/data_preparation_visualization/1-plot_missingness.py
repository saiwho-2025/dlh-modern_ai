#!/usr/bin/env python3
"""a function that visualizes missing values in a dataframe"""
import matplotlib.pyplot as plt


def plot_missingness(df):
    """Visualize missing values in a pandas DataFrame."""
    missing = df.isna()
    row_indices, col_indices = missing.to_numpy().nonzero()

    plt.figure()
    plt.scatter(row_indices, col_indices, marker="|")
    plt.xlabel("Row index")
    plt.ylabel("Columns")
    plt.yticks(range(len(df.columns)), df.columns)
    plt.show()
