#!/usr/bin/env python3
"""
Visualize correlations between continuous numeric features.
"""
import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation_heatmap(df):
    """
    Plot a correlation heatmap for continuous numeric features.
    """
    numeric_df = df.select_dtypes(include="number")

    continuous_columns = [
        col for col in numeric_df.columns
        if numeric_df[col].nunique(dropna=True) > 2
    ]

    corr_matrix = numeric_df[continuous_columns].corr()

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig("Task_9.png")
    plt.show()

    return None
