from pathlib import Path
import pandas as pd

dataset_path = Path(__file__).resolve().parent.parent / "datasets" / "911.csv"

print(f"Reading dataset from:\n{dataset_path}\n")

df = pd.read_csv(dataset_path)

print("Dataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())