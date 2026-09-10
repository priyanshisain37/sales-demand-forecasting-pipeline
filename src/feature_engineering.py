import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load cleaned dataset
df = pd.read_csv("data/cleaned/cleaned_sales.csv")

# Categorical columns
categorical_columns = [
    "State",
    "Market",
    "Market Size",
    "Product Type",
    "Product",
    "Type"
]

# Encode each categorical column
encoder = LabelEncoder()

for column in categorical_columns:
    df[column] = encoder.fit_transform(df[column])

# Save processed dataset
df.to_csv("data/cleaned/processed_sales.csv", index=False)

print("✅ Feature Engineering Completed Successfully!")
print(df.head())