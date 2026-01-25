
import pandas as pd
import numpy as np
from collections import Counter
import json

# Load and analyze the IoT dataset
df = pd.read_csv('IoTData-Raw.csv')

print("Dataset Shape:", df.shape)
print("\n" + "="*80)
print("Column Names and Types:")
print(df.dtypes)
print("\n" + "="*80)
print("First Few Rows:")
print(df.head(10))
print("\n" + "="*80)
print("Dataset Info:")
print(df.info())
print("\n" + "="*80)
print("Statistical Summary:")
print(df.describe())
print("\n" + "="*80)
print("Missing Values:")
print(df.isnull().sum())
print("\n" + "="*80)
print("Unique Values in Categorical Columns:")
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"{col}: {df[col].nunique()} unique values")
        print(f"  Values: {df[col].unique()[:10]}")
