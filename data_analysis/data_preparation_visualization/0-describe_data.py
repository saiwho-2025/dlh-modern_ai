#!/usr/bin/env python3
"""describe data"""

import pandas as pd

df = pd.read_csv('Telco-Customer-Churn.csv')
print("Shape:", df.shape)
print("Dtypes:")
print(df.dtypes)
print("First rows:")
print(df.head())
print("Missing values:")
print(df.isna().sum())
print("Duplicates:", df.duplicated().sum())