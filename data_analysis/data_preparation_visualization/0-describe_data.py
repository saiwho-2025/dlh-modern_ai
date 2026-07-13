#!/usr/bin/env python3
"""describe data"""

import pandas as pd


df = pd.read_csv('Telco-Customer-Churn.csv')
shape = df.shape
data_types = df.dtypes 
head = df.head(n =-5)
missing_count = df.NA.count
duplicates = df.duplicated.count
print [shape, data_types, head, missing_count, duplicates]