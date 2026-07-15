#!/usr/bin/env python3
"""
Write a function that visualizes categorical feature distributions
"""
import matplotlib.pyplot as plt


def plot_categorical_distributions(df, columns_to_plot=None):
    """
    Visualize categorical feature distributions.
    """
    if columns_to_plot is None:
        df = df.drop("Churn", axis=1)
        df = df.select_dtypes(include="object")
        columns_to_plot = df.columns.to_list()
    else:
        if isinstance(columns_to_plot, str):
            columns_to_plot = [columns_to_plot]

    num_plots = len(columns_to_plot)
    n_cols = 3
    n_rows = (num_plots + 2) // 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, col in zip(axes, columns_to_plot):
        counts = df[col].value_counts()
        ax.bar(counts.index, counts.values)
        ax.set_title(col)
        ax.tick_params(axis="x", labelrotation=45)

    for ax in axes[num_plots:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("Task_7.png")
    plt.show()
