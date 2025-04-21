import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix

# ------------------ UI HEADER ------------------
st.set_page_config(page_title="Stock Prediction Dashboard", layout="centered")
st.markdown("# 📈 Stock Price Prediction & Analysis")

# ------------------ MODE SELECTION USING TABS ------------------
tabs = st.tabs([
    " Upload CSV", 
    " Live Stock Tickers (Single)", 
    " Compare Two Tickers", 
    " Sector Comparison & Risk Analysis"
])

# ------------------ Helper Functions ------------------

def calculate_ema(data, period):
    return data['Close'].ewm(span=period, adjust=False).mean()

@st.cache_data
def load_stock_data(ticker):
    try:
        stock_data = yf.download(ticker, start="2020-01-01", end=datetime.now().strftime('%Y-%m-%d'))
        if stock_data.empty:
            raise ValueError(f"No data available for {ticker}")
        stock_data.reset_index(inplace=True)
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = [col[0] for col in stock_data.columns]
        return stock_data
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")
        return pd.DataFrame()

def generate_signal(data):
    data['Short_MA'] = data['Close'].rolling(window=20).mean()
    data['Long_MA'] = data['Close'].rolling(window=50).mean()
    data['Signal'] = 0
    data['Signal'][20:] = np.where(data['Short_MA'][20:] > data['Long_MA'][20:], 1, -1)
    return data

def recommendation(signal):
    if signal == 1:
        return "BUY"
    elif signal == -1:
        return "SELL"
    return "HOLD"

# ------------------ TAB 1: Upload CSV ------------------

with tabs[0]:
    st.subheader(" Upload CSV for Custom Stock Prediction")

    uploaded_file = st.file_uploader("Upload TCS.csv File", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Data Preview", data.head())

        features = ['Prev Close', 'Open', 'High', 'Low', 'VWAP', 'Volume']
        target = 'Close'

        if all(col in data.columns for col in features + [target]):
            X = data[features]
            y = data[target]

            model_type = st.selectbox("Choose a Model", ["Linear Regression", "ID3 Decision Tree", "Naive Bayes"])

            if model_type == "Linear Regression":
                st.subheader("Linear Regression Model")
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                rmse = mean_squared_error(y_test, y_pred) ** 0.5
                st.write(f"RMSE: {rmse:.2f}")

                plt.figure(figsize=(10, 6))
                plt.scatter(y_test, y_pred, alpha=0.5)
                plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
                plt.title("Actual vs Predicted Close Values")
                plt.xlabel("Actual Close")
                plt.ylabel("Predicted Close")
                st.pyplot(plt)

            elif model_type == "ID3 Decision Tree":
                st.subheader("ID3 Decision Tree Classifier")
                y_binned = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile').fit_transform(y.values.reshape(-1, 1)).ravel()

                X_train, X_test, y_train_binned, y_test_binned = train_test_split(X, y_binned, test_size=0.2, random_state=42)
                model = DecisionTreeClassifier(criterion='entropy')
                model.fit(X_train, y_train_binned)
                y_pred = model.predict(X_test)

                st.write(f"Accuracy: {accuracy_score(y_test_binned, y_pred) * 100:.2f}%")

                plt.figure(figsize=(15, 8))
                plot_tree(model, feature_names=features, class_names=["Low", "High"], filled=True)
                st.pyplot(plt)

            elif model_type == "Naive Bayes":
                st.subheader("Naive Bayes Model")
                y_binned = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile').fit_transform(y.values.reshape(-1, 1)).ravel()
                X_train, X_test, y_train_binned, y_test_binned = train_test_split(X, y_binned, test_size=0.2, random_state=42)

                model = GaussianNB()
                model.fit(X_train, y_train_binned)
                y_pred = model.predict(X_test)

                st.write(f"Accuracy: {accuracy_score(y_test_binned, y_pred) * 100:.2f}%")

                report = pd.DataFrame(classification_report(y_test_binned, y_pred, target_names=["Low", "High"], output_dict=True)).T
                st.dataframe(report)

                cm = confusion_matrix(y_test_binned, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred Low", "Pred High"], yticklabels=["Act Low", "Act High"])
                st.pyplot(fig)

# ------------------ TAB 2: Real-Time Dashboard ------------------

with tabs[1]:
    st.subheader("Live Stock Ticker Analysis")
    ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL)", value="AAPL")

    if ticker:
        data = load_stock_data(ticker)
        if not data.empty:
            data = generate_signal(data)
            last_signal = data['Signal'].iloc[-1]
            action = recommendation(last_signal)

            if last_signal == 1:
                st.success(f"Buy Alert for {ticker}")
            elif last_signal == -1:
                st.error(f"Sell Alert for {ticker}")
            else:
                st.warning(f"Hold Signal for {ticker}")

            st.write("### Historical Data Preview")
            st.dataframe(data.tail())

            st.write("### Stock Closing Price Chart")
            fig = px.line(data, x='Date', y='Close', title=f"Closing Price of {ticker}")
            st.plotly_chart(fig)

            st.write("### Exponential Moving Average")
            ema_period = st.slider("Select EMA Period", 5, 50, 20)
            data['EMA'] = calculate_ema(data, ema_period)
            fig_ema = px.line(data, x='Date', y=['Close', 'EMA'], title=f"EMA ({ema_period}) vs Close for {ticker}")
            st.plotly_chart(fig_ema)

            st.write("### Moving Average Crossover")
            fig_ma = px.line(data, x='Date', y=['Close', 'Short_MA', 'Long_MA'], title=f"MA Crossover for {ticker}")
            st.plotly_chart(fig_ma)

            st.write("### Daily Return Distribution")
            returns = data['Close'].pct_change().dropna()
            fig_ret, ax = plt.subplots()
            sns.histplot(returns, bins=50, kde=True, ax=ax)
            ax.set_title(f'Daily Return Distribution for {ticker}')
            st.pyplot(fig_ret)

            st.markdown(f"### AI Recommendation: *{action}*")

# ------------------ TAB 3: Compare Two Tickers ------------------

with tabs[2]:
    st.subheader("🔄 Compare Two Stock Tickers")
    col1, col2 = st.columns(2)
    with col1:
        ticker1 = st.text_input("First Ticker", value="AAPL")
    with col2:
        ticker2 = st.text_input("Second Ticker", value="MSFT")

    if ticker1 and ticker2:
        df1 = load_stock_data(ticker1)
        df2 = load_stock_data(ticker2)

        if not df1.empty and not df2.empty:
            df1 = df1.rename(columns={'Close': f'Close_{ticker1}'})
            df2 = df2.rename(columns={'Close': f'Close_{ticker2}'})
            merged = pd.merge(df1[['Date', f'Close_{ticker1}']], df2[['Date', f'Close_{ticker2}']], on='Date')

            fig = px.line(merged, x='Date', y=[f'Close_{ticker1}', f'Close_{ticker2}'],
                          title=f"Stock Closing Price Comparison: {ticker1} vs {ticker2}")
            st.plotly_chart(fig)

            returns1 = df1.set_index('Date')[f'Close_{ticker1}'].pct_change()
            returns2 = df2.set_index('Date')[f'Close_{ticker2}'].pct_change()

            combined_returns = pd.concat([returns1, returns2], axis=1).dropna()
            combined_returns.columns = [f'{ticker1} Return', f'{ticker2} Return']

            st.write("### 📉 Daily Returns Distribution")
            fig_ret, ax = plt.subplots()
            sns.kdeplot(combined_returns[f'{ticker1} Return'], label=ticker1, ax=ax)
            sns.kdeplot(combined_returns[f'{ticker2} Return'], label=ticker2, ax=ax)
            ax.legend()
            st.pyplot(fig_ret)

# ------------------ TAB 4: Sector Comparison & Risk Analysis ------------------

with tabs[3]:
    st.subheader("🢨 Sector-Wise Heatmap & Risk Metrics")
    sector_tickers = st.text_input("Enter 5-10 tickers from different sectors (comma-separated):", value="AAPL, MSFT, TSLA, JPM, AMZN, NVDA")
    tickers = [t.strip().upper() for t in sector_tickers.split(",") if t.strip()]

    if tickers:
        returns, volatility, sharpe_ratios, sortino_ratios, max_drawdowns = {}, {}, {}, {}, {}
        rf_rate = 0.01

        for ticker in tickers:
            df = load_stock_data(ticker)
            if df.empty:
                continue

            df['Return'] = df['Close'].pct_change()
            df['Negative Return'] = df['Return'].apply(lambda x: x if x < 0 else 0)
            cumulative = (1 + df['Return']).cumprod()
            rolling_max = cumulative.cummax()
            drawdown = (cumulative - rolling_max) / rolling_max

            mean_return = df['Return'].mean()
            std_dev = df['Return'].std()
            downside_std = df['Negative Return'].std()

            returns[ticker] = mean_return * 252
            volatility[ticker] = std_dev * np.sqrt(252)
            sharpe_ratios[ticker] = (returns[ticker] - rf_rate) / volatility[ticker] if volatility[ticker] else 0
            sortino_ratios[ticker] = (returns[ticker] - rf_rate) / downside_std if downside_std else 0
            max_drawdowns[ticker] = drawdown.min()

        metrics_df = pd.DataFrame({
            'Annual Return': returns,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratios,
            'Sortino Ratio': sortino_ratios,
            'Max Drawdown': max_drawdowns
        })

        st.write("### 🔥 Sector-Wise Metrics Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(metrics_df.T, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
        st.pyplot(fig)

        st.write("### 🏆 Top Performing Tickers by Sharpe Ratio")
        top_sharpe = metrics_df.sort_values(by="Sharpe Ratio", ascending=False).head(5)
        st.dataframe(top_sharpe.style.highlight_max(axis=0, color='lightgreen'))

        st.write("### 🌐 Radar Chart for Comparison")
        radar_data = metrics_df[['Annual Return', 'Volatility', 'Sharpe Ratio']]
        radar_fig = px.line_polar(radar_data.reset_index().melt(id_vars='index'), r='value', theta='variable', color='index', line_close=True)
        radar_fig.update_traces(fill='toself')
        st.plotly_chart(radar_fig)

        # ✅ NEW FEATURE: Recommendation Engine Table
        st.write("### 📌 Next Best Action (Buy/Hold/Sell Recommendations)")

        action_data = []
        for ticker in tickers:
            df = load_stock_data(ticker)
            if df.empty or len(df) < 50:
                continue
            df = generate_signal(df)
            latest_signal = df['Signal'].iloc[-1]
            action = recommendation(latest_signal)
            action_data.append({'Ticker': ticker, 'Recommendation': action})

        action_df = pd.DataFrame(action_data)

        st.dataframe(
            action_df.style.applymap(
                lambda val: 'background-color: lightgreen' if val == "BUY"
                else 'background-color: salmon' if val == "SELL"
                else 'background-color: lightyellow',
                subset=['Recommendation']
            )
        )

        st.download_button(
            label="📅 Download Metrics as CSV",
            data=metrics_df.to_csv().encode('utf-8'),
            file_name='sector_metrics.csv',
            mime='text/csv')
