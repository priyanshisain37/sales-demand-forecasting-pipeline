import pandas as pd
import joblib
import os


# ---------------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------------

model = joblib.load(
    "model/sales_forecasting_model.pkl"
)


# ---------------------------------------------------
# LOAD PROCESSED DATA
# ---------------------------------------------------

df = pd.read_csv(
    "data/cleaned/processed_sales.csv"
)


# ---------------------------------------------------
# SELECT SAME FEATURES USED DURING TRAINING
# ---------------------------------------------------

features = [
    "Area Code",
    "State",
    "Market",
    "Market Size",
    "ProductId",
    "Product Type",
    "Product",
    "Type",
    "Year",
    "Month",
    "Day"
]

X = df[features]


# ---------------------------------------------------
# GENERATE PREDICTIONS
# ---------------------------------------------------

df["Predicted_Sales"] = model.predict(X)


# ---------------------------------------------------
# SAVE PREDICTIONS
# ---------------------------------------------------

os.makedirs("output", exist_ok=True)

df.to_csv(
    "output/forecast.csv",
    index=False
)


print("Predictions generated successfully!")

print("\nSample Predictions")
print("-" * 30)

print(
    df[["Sales", "Predicted_Sales"]].head()
)

print("\nForecast saved successfully!")