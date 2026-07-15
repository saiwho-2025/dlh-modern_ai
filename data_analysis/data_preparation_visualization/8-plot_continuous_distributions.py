#!/usr/bin/env python3
"""
Write a function that visualizes
the distributions of continuous numerical features.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_continuous_distributions(df, columns_to_plot=None):
    """
    Visualize continuous numerical feature distributions.
    """
    if columns_to_plot is None:
        df = df.select_dtypes(include="number")
        columns_to_plot = df.columns.to_list()
    else:
        if isinstance(columns_to_plot, str):
            columns_to_plot = [columns_to_plot]

    num_plots = len(columns_to_plot)

    fig, axes = plt.subplots(num_plots, 2, figsize=(10, 3 * num_plots))

    if num_plots == 1:
        axes = axes.reshape(1, -1)

    axes_flat = axes.flatten()

    for i, col in enumerate(columns_to_plot):
        ax_left = axes_flat[2 * i]
        ax_right = axes_flat[2 * i + 1]

        data = df[col].dropna()

        ax_left.hist(
            data,
            bins=30,
            density=True,
            alpha=0.7,
            edgecolor="black"
        )

        kde = stats.gaussian_kde(data)
        x_vals = np.linspace(data.min(), data.max(), 200)

        ax_left.plot(x_vals, kde(x_vals), color="red", ls="--")
        ax_left.set_title(f"{col} Histogram + KDE")

        ax_right.boxplot(data, orientation="horizontal")
        ax_right.set_title(f"{col} Boxplot")

    plt.tight_layout()
    plt.savefig("Task_8.png")
    plt.show()

    return None
