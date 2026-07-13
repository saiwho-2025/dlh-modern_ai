#!/usr/bin/env python3
"""describe data"""

import pandas as pd

df = pd.read_csv('Telco-Customer-Churn.csv')
shape = df.shape
data_types = df.dtypes
head = df.head()
missing_count = df.isna().sum()
duplicates = df.duplicated().sum()
print(f"Shape: {shape}\nDtypes:\n{data_types}\nFirst rows:\n{head}")
print(f"Missing values:\n{missing_count}\nDuplicates: {duplicates}")