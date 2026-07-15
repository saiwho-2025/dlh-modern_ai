#!/usr/bin/env python3
"""
Creating a function visualizing correlations between continuous
numeric features.
"""
import seaborn as sns
import matplotlib.pyplot as plt


def plot_correlation_heatmap(df):
    """
    Visualize correlations between numeric features.
    """
    plt.figure(figsize=(6, 5))

    df_plot = df.select_dtypes(include=["number"])
    corr_matrix = df_plot.corr()

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    plt.title("Correlation Matrix")
    plt.show()

    return None
