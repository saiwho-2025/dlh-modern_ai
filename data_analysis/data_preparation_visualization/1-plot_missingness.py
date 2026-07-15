#!/usr/bin/env python3
"""a function that visualizes missing values in a dataframe"""
import matplotlib.pyplot as plt
import numpy as np


def plot_missingness(df):
    """Visualize missing values in a pandas DataFrame."""
    missing = df.isna().to_numpy()
    row_indices, col_indices = np.where(missing)
    plt.scatter(row_indices, col_indices, marker="|")
    plt.yticks(np.arange(len(df.columns)), df.columns)
    plt.title("Missingness Plot")
    plt.show()
