#!/usr/bin/env python3
"""Function visualizing churn rates per category."""
import matplotlib.pyplot as plt


def plot_categorical_vs_churn(df, col):
    """Visualize churn rates per category."""
    churn_rate = df.groupby(col)["Churn"].value_counts(normalize=True)
    churn_rate_yes = churn_rate[:, "Yes"]

    labels = churn_rate_yes.index
    values = churn_rate_yes.values

    plt.figure(figsize=(12, 8))
    plt.bar(labels, values)
    plt.ylabel("Churn Rate")
    plt.title(f"Churn Rate by {col}")
    plt.xticks(rotation=45)
    plt.show()

    return None
