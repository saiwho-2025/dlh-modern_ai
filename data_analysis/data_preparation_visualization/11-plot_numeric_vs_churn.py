#!/usr/bin/env python3
"""Function comparing continuous features against Churn."""
import matplotlib.pyplot as plt


def plot_numeric_vs_churn(df, col):
    """Compare continuous numeric feature distributions by churn."""
    col_no = df[df["Churn"] == "No"][col]
    col_yes = df[df["Churn"] == "Yes"][col]

    plt.figure(figsize=(12, 8))

    plt.hist([col_no, col_yes], bins=30, label=["No", "Yes"])

    plt.title(f"{col} Distribution by Churn")
    plt.xlabel(col)
    plt.legend(title="Churn")

    plt.show()

    return None
