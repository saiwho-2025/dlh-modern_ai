#!/usr/bin/env python3
"""a function handles missing values"""


def clean_total_charges(df, method='drop'):
    """the method handles different missing values in PandsDataFrame"""
[copy_files] Filed copied: 1-main.py
[copy_files] Filed copied: Telco-Customer-Churn.csv
[compare] Command to run:
./1-main.py 2>/dev/null
su student_jail -c 'timeout 30 bash -c '"'"'./1-main.py 2>/dev/null'"'"''
[compare] Return code: 0
[compare] Student stdout:
Series([], Name: TotalCharges, dtype: float64)
[compare] Student stdout length: 47
[compare] Student stderr:
[compare] Student stderr length: 0
[compare] Desired stdout:
488 1397.475
753 1397.475
936 1397.475
1082 1397.475
1340 1397.475
3331 1397.475
3826 1397.475
4380 1397.475
5218 1397.475
6670 1397.475
6754 1397.475
Name: TotalCharges, dtype: float64
[compare] Desired stdout length: 222
Worked for 26s

Replace your function with this version. It uses no imports and does not drop rows when method='median'.

def clean_total_charges(df, method='drop'):
    def to_float(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return float("nan")

    df["TotalCharges"] = df["TotalCharges"].apply(to_float)

    if method == "drop":
        df = df.dropna(subset=["TotalCharges"])

    elif method == "median":
        median_value = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(median_value)

    elif method == "impute":
        missing = df["TotalCharges"].isna()
        df.loc[missing, "TotalCharges"] = (
            df.loc[missing, "MonthlyCharges"] * df.loc[missing, "tenure"]
        )

    else:
        raise ValueError("method must be 'drop', 'median', or 'impute'")

    return df