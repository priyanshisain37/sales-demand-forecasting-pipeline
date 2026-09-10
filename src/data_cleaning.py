import pandas as pd
import os

# Create output folder
os.makedirs("data/cleaned", exist_ok=True)

# Load dataset
df = pd.read_csv("data/raw/sales.csv")

print("Original Shape:", df.shape)

# Remove duplicate rows
df = df.drop_duplicates()

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values
df = df.ffill()

# Convert Date column to date
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%y %H:%M:%S")

# Create new features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

# Save cleaned dataset
df.to_csv("data/cleaned/cleaned_sales.csv", index=False)

print("\nCleaned Shape:", df.shape)
print("\nCleaned dataset saved successfully!")