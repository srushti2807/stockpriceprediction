import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
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

# Function to calculate RSI
def calculate_rsi(data, period=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate VWAP
def calculate_vwap(data):
    return (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()

# Streamlit App
st.title("📈 Stock Price Prediction & Buy/Sell Recommendation")

# Function to load stock data
@st.cache_data
def load_stock_data(ticker):
    stock_data = yf.download(ticker, start="2020-01-01", end=datetime.now().strftime('%Y-%m-%d'))
    return stock_data

# User input for stock symbol
ticker = st.text_input("Enter Stock Ticker Symbol:", "AAPL")

# Load stock data
data = load_stock_data(ticker)

if not data.empty:
    st.subheader(f"Data Preview for {ticker}")
    st.write(data.tail())

    # Creating additional features for tomorrow's price prediction
    data['Tomorrow_Close'] = data['Close'].shift(-1)
    
    # Drop rows with NaN values
    data.dropna(inplace=True)

    # Calculate Technical Indicators
    data['EMA'] = calculate_ema(data, period=20)
    data['Middle Band'], data['Upper Band'], data['Lower Band'] = calculate_bollinger_bands(data)
    data['RSI'] = calculate_rsi(data, period=14)
    data['VWAP'] = calculate_vwap(data)

    # Feature Selection
    features = ['Close', 'Open', 'High', 'Low', 'Volume', 'EMA', 'RSI', 'VWAP']
    target_close = 'Tomorrow_Close'

    X = data[features]
    y_close = data[target_close]

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Model Selection
    model_type = st.selectbox("Choose a model:", ["Linear Regression", "Random Forest"])

    # Splitting the Data
    X_train, X_test, y_train_close, y_test_close = train_test_split(X, y_close, test_size=0.2, random_state=42)

    # Model Training
    next_day_close_pred = None

    if model_type == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train_close)
    y_pred_close = model.predict(X_test)

    # Model Evaluation
    mse_close = mean_squared_error(y_test_close, y_pred_close)
    rmse_close = np.sqrt(mse_close)
    st.write(f"📉 RMSE for {model_type}: **{rmse_close:.2f}**")

    # Predicting Tomorrow's Close
    last_row = data[features].iloc[-1].values.reshape(1, -1)
    next_day_close_pred = float(model.predict(last_row)[0])  # Ensure single float value
    today_close = float(data['Close'].iloc[-1])  # Ensure single float value

    st.write(f"🔮 Predicted Next Day's Closing Price: **{next_day_close_pred:.2f}**")

    # Buy/Sell Recommendation
    recommendation = "🔼 Buy" if next_day_close_pred > today_close else "🔽 Sell"
    st.subheader(f"💡 Recommendation: {recommendation}")
    st.write(f"📌 Today's Close: **{today_close:.2f}** | Predicted: **{next_day_close_pred:.2f}**")

    # Plot Actual vs Predicted Values
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test_close, y_pred_close, alpha=0.5, color='blue')
    plt.plot([min(y_test_close), max(y_test_close)], [min(y_test_close), max(y_test_close)], color='red', linestyle="--")
    plt.title(f'Actual vs Predicted Tomorrow\'s Close for {ticker}')
    plt.xlabel('Actual Close')
    plt.ylabel('Predicted Close')
    st.pyplot(plt)

    # Plot Recent Stock Prices & Prediction
    plt.figure(figsize=(10, 5))
    plt.plot(data['Close'][-30:], label='Last 30 Days Close', color='blue')
    plt.axhline(y=next_day_close_pred, color='red', linestyle='--', label='Predicted Close')
    plt.title(f'{ticker} - Last 30 Days Closing Prices & Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

    # Stock Price Chart
    st.subheader(f"📊 {ticker} Closing Price Chart")
    st.line_chart(data['Close'])

else:
    st.error("❌ No data available for the given ticker symbol.")
