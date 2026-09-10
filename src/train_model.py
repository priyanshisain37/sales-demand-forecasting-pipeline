import pandas as pd
import joblib
import os

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# ---------------------------------------------------
# LOAD PROCESSED DATA
# ---------------------------------------------------

df = pd.read_csv("data/cleaned/processed_sales.csv")


# ---------------------------------------------------
# CONVERT DATE AND SORT CHRONOLOGICALLY
# ---------------------------------------------------

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df = df.sort_values("Date").reset_index(drop=True)


# ---------------------------------------------------
# SELECT FEATURES
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
y = df["Sales"]


# ---------------------------------------------------
# TIME-BASED TRAIN TEST SPLIT
# ---------------------------------------------------

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("Training Data:", X_train.shape)
print("Testing Data :", X_test.shape)


# ---------------------------------------------------
# TRAIN XGBOOST MODEL
# ---------------------------------------------------

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)


# ---------------------------------------------------
# MODEL PREDICTION
# ---------------------------------------------------

y_pred = model.predict(X_test)


# ---------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5


print("\nModel Performance")
print("-" * 30)

print("R² Score :", round(r2, 4))
print("MAE      :", round(mae, 4))
print("RMSE     :", round(rmse, 4))


# ---------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------

os.makedirs("model", exist_ok=True)

joblib.dump(
    model,
    "model/sales_forecasting_model.pkl"
)

print("\n✅ Model trained and saved successfully!")