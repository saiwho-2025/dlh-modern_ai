#!/usr/bin/env python3
"""a function visualize churn class distribution"""


def plot_churn_distribution(df):
    """the plot"""
    counts = df["Churn"].value_counts()
    counts.plot(kind="bar", color=["skyblue", "salmon"])
