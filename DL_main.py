import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import load_model
from datetime import date, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil.relativedelta import relativedelta

# --------------------
# Constants
# --------------------
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
MODEL_PATH = "m.h5"

# --------------------
# Streamlit UI
# --------------------
st.title("📈 Google Stock Price Prediction")
st.write("Explore Google stock with EDA, model evaluation, and predictions.")

ticker = "GOOGL"

# --------------------
# Load Data
# --------------------
data = yf.download(ticker, START, TODAY, auto_adjust=False)
data.reset_index(inplace=True)

# Fix MultiIndex columns if needed
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] for col in data.columns]

df = data[['Close']]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Load model
model = load_model(MODEL_PATH)

# --------------------
# Tabs
# --------------------
tab1, tab2, tab3 = st.tabs(["📊 Exploratory Data Analysis", "📈 Model Evaluation", "🔮 Prediction"])

# ==============================================================  
# 1. Exploratory Data Analysis  
# ==============================================================  
with tab1:
    st.subheader("Dataset Overview")

    start_date = data["Date"].min()
    end_date = data["Date"].max()
    diff = relativedelta(end_date, start_date)
    time_range = f"{diff.years} Years {diff.months} Months {diff.days} Days"

    st.write("**Shape of Dataset:**", data.shape)
    # st.write("**Columns:**", list(data.columns))
    st.write("**Time Range:**", f"{start_date.date()} → {end_date.date()}  ({time_range})")



    # st.write("**First 5 Rows**")
    # st.write(data.head())
    st.write("**Summary Statistics**")
    st.dataframe(data.describe())


    # ---------------- Outlier Detection ---------------- #
    Q1 = data['Close'].quantile(0.25)
    Q3 = data['Close'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data['Close'] < lower_bound) | (data['Close'] > upper_bound)]

    st.write("📌 Outlier Detection (Close Price)")
    st.write(f"Detected **{len(outliers)} outliers** in Close price.")

    if not outliers.empty:
        st.dataframe(outliers[['Date', 'Close']].head(10))  # show top 10 outliers

    # Boxplot for Close Prices (with outliers)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x=data['Close'], ax=ax)
    ax.set_title("Boxplot of Close Prices (Outliers Highlighted)")
    st.pyplot(fig)


    # Distribution of Closing Price
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data['Close'], bins=50, kde=True, ax=ax)
    ax.set_title("Distribution of Closing Prices")
    st.pyplot(fig)

    # Closing Price Over Time
    st.write("**Closing Price Over Time**")
    st.line_chart(data.set_index("Date")["Close"])

    # Volume Over Time
    st.write("**Trading Volume Over Time**")
    st.line_chart(data.set_index("Date")["Volume"])

    # Correlation Heatmap
    st.write("**Correlation Between Features**")
    corr = data[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    st.write("**Correlation Insights:**")
    st.markdown("""
    - `Close Price` is usually highly correlated with `Open`, `High`, and `Low`.
    - `Volume` often has weaker correlation with prices.
    """)

# ==============================================================  
# 2. Model Evaluation  
# ==============================================================  
with tab2:
    st.subheader("📈 LSTM Model Evaluation")

    look_back = 60
    split = int(len(scaled_data) * 0.8)

    if len(scaled_data) > look_back:
        test_data = scaled_data[split - look_back:]

        # Create test sequences
        X_test, y_test = [], []
        for i in range(look_back, len(test_data)):
            X_test.append(test_data[i - look_back:i, 0])
            y_test.append(test_data[i, 0])

        X_test, y_test = np.array(X_test), np.array(y_test)

        if X_test.size > 0:
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

            with st.spinner("⏳ Evaluating model..."):
                try:
                    y_pred_scaled = model.predict(X_test, verbose=0)
                    y_pred = scaler.inverse_transform(y_pred_scaled)
                    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

                    # Metrics
                    rmse = round(np.sqrt(mean_squared_error(y_test_actual, y_pred)), 2)
                    r2 = round(r2_score(y_test_actual, y_pred), 2)
                    mape = round(np.mean(np.abs((y_test_actual - y_pred) / y_test_actual)) * 100, 2)

                    # KPI Cards
                    st.markdown(
                        """
                        <style>
                        .card {
                            background-color: #f8f9fa;
                            padding: 20px;
                            border-radius: 12px;
                            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
                            text-align: center;
                            font-family: "Arial", sans-serif;
                        }
                        .card h3 {
                            margin: 0;
                            font-size: 18px;
                            color: #333;
                        }
                        .card p {
                            margin: 5px 0 0;
                            font-size: 18px;
                            font-weight: bold;
                            color: #0073e6;
                        }
                        </style>
                        """, unsafe_allow_html=True
                    )

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"<div class='card'><h3>R² Score</h3><p>{r2}</p></div>", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"<div class='card'><h3>RMSE</h3><p>{rmse}</p></div>", unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"<div class='card'><h3>MAPE (%)</h3><p>{mape}</p></div>", unsafe_allow_html=True)

                    # ====================== NEW GRAPH ====================== #
                    st.subheader("📊 Actual vs Predicted Stock Prices")

                    test_dates = data['Date'][split:].reset_index(drop=True)  # Dates for test set

                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(test_dates[:len(y_test_actual)], y_test_actual, label="Actual Prices", color="blue")
                    ax.plot(test_dates[:len(y_pred)], y_pred, label="Predicted Prices", color="orange")
                    ax.set_title("Actual vs Predicted Google Stock Prices")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Price")
                    ax.legend()
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    # ========================================================= #

                except Exception as e:
                    st.error(f"Model prediction failed: {e}")
        else:
            st.error("❌ Not enough data for evaluation.")
    else:
        st.error("❌ Not enough historical data for evaluation.")

# ==============================================================  
# 3. Prediction  
# ==============================================================  
with tab3:
    st.subheader("🔮 Stock Price Prediction")
    user_date = st.date_input(
        "Pick a date",
        value=date.today(),
        min_value=date(2020, 1, 1),
        max_value=date.today() + timedelta(days=30)
    )

    if st.button("Predict Price") and user_date is not None:
        with st.spinner("⏳ Processing prediction..."):

            predicted_price = None  

            if user_date > date.today():  # Future prediction
                last_60 = df[-60:].values
                scaled_last_60 = scaler.transform(last_60)
                days_ahead = (user_date - date.today()).days
                seq = scaled_last_60

                for _ in range(days_ahead):
                    pred = model.predict(seq.reshape(1, seq.shape[0], 1), verbose=0)
                    seq = np.append(seq[1:], pred).reshape(-1, 1)
                    predicted_price = scaler.inverse_transform(pred)[0][0]

            else:  # Past/Present (only predicted values, no actual)
                idx_match = data[data['Date'] == pd.to_datetime(user_date)]
                if not idx_match.empty:
                    idx = idx_match.index[0]
                    if idx >= 60:
                        seq = scaled_data[idx-60:idx]
                        seq = np.reshape(seq, (1, seq.shape[0], 1))
                        pred = model.predict(seq, verbose=0)
                        predicted_price = scaler.inverse_transform(pred)[0][0]

            # Show result with highlight (like st.warning)
            if predicted_price is not None:
                st.success(f"📌 Prediction for {user_date}:  **${predicted_price:.2f}**")
            else:
                st.warning("⚠️ Prediction not available (not enough historical data).")
