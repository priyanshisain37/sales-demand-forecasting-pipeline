import pandas as pd

df = pd.read_csv("data/raw/sales.csv")

print(df.head())
print("\nShape:", df.shape)
print("\nColumns:")
print(df.columns)
print("\nMissing Values:")
print(df.isnull().sum())