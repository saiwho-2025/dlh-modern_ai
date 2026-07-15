#!/usr/bin/env python3
"""a function visualize churn class distribution"""
import matplotlib.pyplot as plt


def plot_churn_distribution(df):
    """Visualize churn class distribution."""
    churn_counts = df["Churn"].value_counts()
    colors = [
        "skyblue" if value == "No" else "salmon"
        for value in churn_counts.index
    ]

    plt.bar(churn_counts.index, churn_counts.values, color=colors)
    plt.xlabel("Churn")
    plt.ylabel("Count")
    plt.title("Churn Distribution")
    plt.show()
