# utils.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

def load_stock_data(ticker):
    try:
        stock_data = yf.download(ticker, start="2020-01-01", end=datetime.now().strftime('%Y-%m-%d'))
        stock_data.reset_index(inplace=True)
        return stock_data
    except:
        return pd.DataFrame()

def calculate_ema(data, period):
    return data['Close'].ewm(span=period, adjust=False).mean()

def generate_signal(data):
    data['Short_MA'] = data['Close'].rolling(window=20).mean()
    data['Long_MA'] = data['Close'].rolling(window=50).mean()
    data['Signal'] = np.where(data['Short_MA'] > data['Long_MA'], 1, -1)
    return data

def recommendation(signal):
    if signal == 1:
        return "BUY"
    elif signal == -1:
        return "SELL"
    return "HOLD"
