import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

prediction_df = pd.read_csv("output/forecast.csv")
cleaned_df = pd.read_csv("data/cleaned/cleaned_sales.csv")

# Add original categorical columns for filtering
for column in ["State", "Market", "Product"]:
    if column in cleaned_df.columns:
        prediction_df[column] = cleaned_df[column]


# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.title("📊 Cloud-Based Sales Forecasting Dashboard")

st.write(
    "Sales prediction and analytics dashboard powered by "
    "Python, XGBoost, and Amazon S3."
)

st.divider()


# ---------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------

st.sidebar.header("🔎 Dashboard Filters")

states = sorted(prediction_df["State"].dropna().unique())
markets = sorted(prediction_df["Market"].dropna().unique())
products = sorted(prediction_df["Product"].dropna().unique())

selected_state = st.sidebar.multiselect(
    "Select State",
    states,
    default=[]
)

selected_market = st.sidebar.multiselect(
    "Select Market",
    markets,
    default=[]
)

selected_product = st.sidebar.multiselect(
    "Select Product",
    products,
    default=[]
)


# ---------------------------------------------------
# APPLY FILTERS
# ---------------------------------------------------

filtered_df = prediction_df.copy()

if selected_state:
    filtered_df = filtered_df[
        filtered_df["State"].isin(selected_state)
    ]

if selected_market:
    filtered_df = filtered_df[
        filtered_df["Market"].isin(selected_market)
    ]

if selected_product:
    filtered_df = filtered_df[
        filtered_df["Product"].isin(selected_product)
    ]


# ---------------------------------------------------
# KPI CALCULATIONS
# ---------------------------------------------------

total_actual = filtered_df["Sales"].sum()
total_predicted = filtered_df["Predicted_Sales"].sum()

average_actual = filtered_df["Sales"].mean()
average_predicted = filtered_df["Predicted_Sales"].mean()

prediction_difference = total_actual - total_predicted


# ---------------------------------------------------
# KPI CARDS
# ---------------------------------------------------

st.subheader("📌 Sales Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Actual Sales",
    f"{total_actual:,.2f}"
)

col2.metric(
    "Total Predicted Sales",
    f"{total_predicted:,.2f}"
)

col3.metric(
    "Average Actual Sales",
    f"{average_actual:,.2f}"
)

col4.metric(
    "Prediction Difference",
    f"{prediction_difference:,.2f}"
)


st.divider()


# ---------------------------------------------------
# ACTUAL VS PREDICTED
# ---------------------------------------------------

st.subheader("📈 Actual vs Predicted Sales")

chart_data = filtered_df[
    ["Sales", "Predicted_Sales"]
].head(100)

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    chart_data["Sales"].values,
    label="Actual Sales"
)

ax.plot(
    chart_data["Predicted_Sales"].values,
    label="Predicted Sales"
)

ax.set_xlabel("Record")
ax.set_ylabel("Sales")
ax.set_title("Actual vs Predicted Sales")

ax.legend()

st.pyplot(fig)


st.divider()


# ---------------------------------------------------
# SALES TREND
# ---------------------------------------------------

st.subheader("📊 Sales Trend")

if "Date" in cleaned_df.columns:

    trend_df = cleaned_df.copy()

    trend_df["Date"] = pd.to_datetime(
        trend_df["Date"],
        errors="coerce"
    )

    trend_df = trend_df.sort_values("Date")

    daily_sales = (
        trend_df
        .groupby("Date")["Sales"]
        .sum()
        .reset_index()
    )

    fig2, ax2 = plt.subplots(figsize=(12, 5))

    ax2.plot(
        daily_sales["Date"],
        daily_sales["Sales"]
    )

    ax2.set_xlabel("Date")
    ax2.set_ylabel("Sales")
    ax2.set_title("Sales Trend Over Time")

    plt.xticks(rotation=45)

    st.pyplot(fig2)


st.divider()


# ---------------------------------------------------
# PREDICTION ERROR
# ---------------------------------------------------

st.subheader("📉 Prediction Error Analysis")

filtered_df["Prediction_Error"] = (
    filtered_df["Sales"]
    - filtered_df["Predicted_Sales"]
)

error_col1, error_col2, error_col3 = st.columns(3)

error_col1.metric(
    "Average Error",
    f"{filtered_df['Prediction_Error'].mean():,.2f}"
)

error_col2.metric(
    "Maximum Error",
    f"{filtered_df['Prediction_Error'].abs().max():,.2f}"
)

error_col3.metric(
    "Mean Absolute Error",
    f"{filtered_df['Prediction_Error'].abs().mean():,.2f}"
)


st.divider()


# ---------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------

st.subheader("🤖 XGBoost Model Performance")

try:
    model = joblib.load("model/sales_forecasting_model.pkl")
    processed_df = pd.read_csv("data/cleaned/processed_sales.csv")

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

    processed_df["Date"] = pd.to_datetime(
        processed_df["Date"],
        errors="coerce"
    )

    processed_df = processed_df.sort_values("Date").reset_index(drop=True)

    X = processed_df[features]
    y = processed_df["Sales"]

    split_index = int(len(processed_df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    model_col1, model_col2, model_col3 = st.columns(3)

    model_col1.metric(
        "R² Score",
        f"{r2:.4f}"
    )

    model_col2.metric(
        "MAE",
        f"{mae:.4f}"
    )

    model_col3.metric(
        "RMSE",
        f"{rmse:.4f}"
    )

    st.info(
        "The model was evaluated using an 80:20 chronological "
        "train-test split to avoid future-data leakage."
    )

except Exception as e:
    st.warning(
        f"Model performance could not be loaded: {e}"
    )


# ---------------------------------------------------
# PREDICTION TABLE
# ---------------------------------------------------

st.subheader("📋 Prediction Results")

display_columns = [
    "State",
    "Market",
    "Product",
    "Sales",
    "Predicted_Sales"
]

available_columns = [
    column
    for column in display_columns
    if column in filtered_df.columns
]

st.dataframe(
    filtered_df[available_columns].head(50),
    width="stretch"
)


st.divider()


# ---------------------------------------------------
# AWS PIPELINE INFORMATION
# ---------------------------------------------------

st.subheader("☁️ AWS Data Engineering Pipeline")

st.markdown(
    """
    **Amazon S3 → Python/Boto3 → Data Cleaning → Feature Engineering
    → XGBoost → Sales Prediction → Amazon S3**

    **S3 Bucket:** `sales-demand-forecasting-pipeline-2026`

    **Data Flow:**

    `raw-data/sales.csv`

    ↓

    `cleaned-data/cleaned_sales.csv`

    ↓

    `processed_sales.csv`

    ↓

    `XGBoost Model`

    ↓

    `predictions/forecast.csv`
    """
)


st.divider()

st.caption(
    "Cloud-Based Sales Forecasting using AWS S3, Python and XGBoost"
)