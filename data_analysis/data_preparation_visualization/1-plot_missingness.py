#!/usr/bin/env python3
"""
This module visualizes missing values in a DataFrame.
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_missingness(df):
    """
    Visualize missing values in a DataFrame using a scatter plot.
    """
    plt.figure(figsize=(12, 8))

    missing_rows, missing_cols = np.where(df.isna())

    plt.scatter(missing_rows, missing_cols, marker="|")

    plt.yticks(
        ticks=np.arange(len(df.columns)),
        labels=df.columns
    )

    plt.title("Missingness Plot")

    plt.tight_layout()
    plt.show()

    return None
