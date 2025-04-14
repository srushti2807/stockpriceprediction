# test_utils.py
import pandas as pd
import numpy as np
from utils import load_stock_data, calculate_ema, generate_signal, recommendation

def test_load_stock_data():
    df = load_stock_data("AAPL")
    assert not df.empty
    assert "Close" in df.columns

def test_calculate_ema():
    data = pd.DataFrame({"Close": np.random.rand(50)})
    ema = calculate_ema(data, 10)
    assert len(ema) == 50
    assert isinstance(ema.iloc[-1], float)

def test_generate_signal():
    data = pd.DataFrame({"Close": np.random.rand(100)})
    data = generate_signal(data)
    assert "Signal" in data.columns
    assert set(data["Signal"].dropna().unique()).issubset({1, -1})

def test_recommendation():
    assert recommendation(1) == "BUY"
    assert recommendation(-1) == "SELL"
    assert recommendation(0) == "HOLD"
