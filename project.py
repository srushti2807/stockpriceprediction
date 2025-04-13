import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import matplotlib.pyplot as plt
from datetime import datetime

# Function to calculate EMA
def calculate_ema(data, period):
    return data['Close'].ewm(span=period, adjust=False).mean()

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    middle_band = data['Close'].rolling(window=window).mean()
    std_dev = data['Close'].rolling(window=window).std()
    upper_band = middle_band + (std_dev * num_std_dev)
    lower_band = middle_band - (std_dev * num_std_dev)
    return middle_band, upper_band, lower_band

# Streamlit App Title
st.title("📈 Multi-Stock Price Prediction & Analysis")

# Function to load stock data
@st.cache_data
def load_stock_data(ticker):
    try:
        stock_data = yf.download(ticker, start="2020-01-01", end=datetime.now().strftime('%Y-%m-%d'))
        if stock_data.empty:
            raise ValueError(f"❌ No data available for {ticker}")
        return stock_data
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")
        return pd.DataFrame()

# Input for multiple stock tickers
tickers = st.text_input("Enter Stock Ticker Symbols (comma-separated):", value="AAPL, MSFT")

# Load and analyze each stock
if tickers:
    ticker_list = [ticker.strip() for ticker in tickers.split(",")]

    for ticker in ticker_list:
        st.subheader(f"📊 Analysis for {ticker}")

        # Load stock data
        data = load_stock_data(ticker)
        if data.empty:
            continue  # Skip if no data

        st.write(f"Data Preview for {ticker}")
        st.write(data.tail())

        # Feature Engineering
        data['Tomorrow_Close'] = data['Close'].shift(-1)
        data.dropna(inplace=True)  # Remove last row where Tomorrow_Close is NaN
        data['EMA'] = calculate_ema(data, period=20)
        data['Middle Band'], data['Upper Band'], data['Lower Band'] = calculate_bollinger_bands(data)

        # Feature Selection
        features = ['Close', 'Open', 'High', 'Low', 'Volume', 'EMA']
        target_close = 'Tomorrow_Close'

        X = data[features]
        y_close = data[target_close]

        # User selects model type
        model_type = st.selectbox(f"Choose a model for {ticker}:", ["Linear Regression", "Naive Bayes", "Random Forest"])

        # Train-Test Split
        X_train, X_test, y_train_close, y_test_close = train_test_split(X, y_close, test_size=0.2, random_state=42)

        # Variable to store next-day prediction
        next_day_close_pred = None

        # **Linear Regression**
        if model_type == "Linear Regression":
            st.subheader(f"📈 Linear Regression Model for {ticker}")
            model = LinearRegression()
            model.fit(X_train, y_train_close)
            y_pred_close = model.predict(X_test)
            mse_close = mean_squared_error(y_test_close, y_pred_close)
            rmse_close = mse_close ** 0.5
            st.write(f"✅ RMSE: **{rmse_close:.2f}**")

            # Predicting next day's close
            last_row = X.iloc[-1].values.reshape(1, -1)
            next_day_close_pred = float(model.predict(last_row)[0])
            st.write(f"🔮 Predicted Next Day Close: **{next_day_close_pred:.2f}**")

            # Scatter Plot
            plt.figure(figsize=(10, 5))
            plt.scatter(y_test_close, y_pred_close, color='blue', alpha=0.5)
            plt.plot([min(y_test_close), max(y_test_close)], [min(y_test_close), max(y_test_close)], color='red', linestyle="--")
            plt.xlabel('Actual Close')
            plt.ylabel('Predicted Close')
            plt.title(f'Actual vs Predicted Close Price for {ticker}')
            st.pyplot(plt)

        # **Naive Bayes**
        elif model_type == "Naive Bayes":
            st.subheader(f"📊 Naive Bayes Model for {ticker}")
            binner_close = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile')
            y_binned_close = binner_close.fit_transform(y_close.values.reshape(-1, 1)).ravel()
            X_train, X_test, y_train_binned_close, y_test_binned_close = train_test_split(X, y_binned_close, test_size=0.2, random_state=42)

            model = GaussianNB()
            model.fit(X_train, y_train_binned_close)
            y_pred_binned_close = model.predict(X_test)
            accuracy_close = accuracy_score(y_test_binned_close, y_pred_binned_close)
            st.write(f"✅ Accuracy: **{accuracy_close * 100:.2f}%**")
            st.write("📄 Classification Report:")
            st.text(classification_report(y_test_binned_close, y_pred_binned_close, target_names=["Low", "High"]))

        # **Random Forest Regressor**
        elif model_type == "Random Forest":
            st.subheader(f"🌲 Random Forest Regressor Model for {ticker}")
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train_close)
            y_pred_close = model.predict(X_test)
            mse_close = mean_squared_error(y_test_close, y_pred_close)
            rmse_close = mse_close ** 0.5
            st.write(f"✅ RMSE: **{rmse_close:.2f}**")

            # Predicting next day's close
            last_row = X.iloc[-1].values.reshape(1, -1)
            next_day_close_pred = float(model.predict(last_row)[0])
            st.write(f"🔮 Predicted Next Day Close: **{next_day_close_pred:.2f}**")

            # Scatter Plot
            plt.figure(figsize=(10, 5))
            plt.scatter(y_test_close, y_pred_close, color='green', alpha=0.5)
            plt.plot([min(y_test_close), max(y_test_close)], [min(y_test_close), max(y_test_close)], color='red', linestyle="--")
            plt.xlabel('Actual Close')
            plt.ylabel('Predicted Close')
            plt.title(f'Actual vs Predicted Close Price for {ticker}')
            st.pyplot(plt)

        # Display last 10 stock prices with prediction
        st.subheader(f"📉 Last 10 Prices & Prediction for {ticker}")
        display_data = data[['Close', 'Tomorrow_Close', 'EMA']].tail(10).copy()
        display_data['Predicted_Tomorrow_Close'] = next_day_close_pred
        st.write(display_data)

        # Stock Price Chart
        st.subheader(f"📊 {ticker} Closing Price Chart")
        st.line_chart(data['Close'])

else:
    st.warning("⚠️ Please enter valid stock ticker symbols.")
