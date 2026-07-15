#!/usr/bin/env python3
"""a function visualize churn class distribution"""
import matplotlib.pyplot as plt


def plot_churn_distribution(df):
    """Generate a bar plot of Churn value counts."""
    plt.figure(figsize=(12, 8))

    counts = df["Churn"].value_counts().reindex(["No", "Yes"])

    plt.bar(counts.index, counts.values, color=["skyblue", "salmon"])
    plt.ylabel("Count")
    plt.title("Churn Distribution")
    plt.show()

    return None
