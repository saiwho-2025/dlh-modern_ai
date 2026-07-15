#!/usr/bin/env python3
"""a function that visualizes missing values in a dataframe"""
from matplotlib import pyplot as plt
import numpy as np


def plot_missingness(df):
    """Visualize missing values in a pandas DataFrame."""
    missing = df.isna().to_numpy()
    row_indices, col_indices = np.where(missing)
    plt.figure()
    plt.scatter(row_indices, col_indices, marker="|")
    plt.xlabel("Row index")
    plt.ylabel("Columns")
    plt.yticks(range(len(df.columns)), df.columns)
    plt.show()
