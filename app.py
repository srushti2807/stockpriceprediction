from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from matplotlib.figure import Figure
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ------------------ Helper Functions ------------------

def calculate_ema(data, period):
    return data['Close'].ewm(span=period, adjust=False).mean()

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
        print(f"Error loading data for {ticker}: {e}")
        return pd.DataFrame()

def generate_signal(data):
    data['Short_MA'] = data['Close'].rolling(window=20).mean()
    data['Long_MA'] = data['Close'].rolling(window=50).mean()
    data['Signal'] = 0
    data.loc[20:, 'Signal'] = np.where(data['Short_MA'][20:] > data['Long_MA'][20:], 1, -1)
    return data

def recommendation(signal):
    if isinstance(signal, (np.ndarray, pd.Series)):
        signal = signal[-1]  # or use signal.iloc[-1] if it's a Series
    if signal == 1:
        return "Buy"
    elif signal == -1:
        return "Sell"
    else:
        return "Hold"

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

def create_plotly_json(fig):
    return json.dumps(fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)

# ------------------ Routes ------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_csv', methods=['GET', 'POST'])
def upload_csv():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload_csv.html', error="No file part")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('upload_csv.html', error="No selected file")
        
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the uploaded file
            data = pd.read_csv(filepath)
            data_preview = data.head().to_html(classes='table table-striped')
            
            features = ['Prev Close', 'Open', 'High', 'Low', 'VWAP', 'Volume']
            target = 'Close'
            
            if all(col in data.columns for col in features + [target]):
                return render_template('upload_csv.html', 
                                      data_preview=data_preview, 
                                      filename=filename,
                                      has_required_columns=True)
            else:
                return render_template('upload_csv.html', 
                                      data_preview=data_preview,
                                      error="CSV does not have required columns")
    
    return render_template('upload_csv.html')

@app.route('/analyze_csv', methods=['POST'])
def analyze_csv():
    filename = request.form.get('filename')
    model_type = request.form.get('model_type')
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    data = pd.read_csv(filepath)
    
    features = ['Prev Close', 'Open', 'High', 'Low', 'VWAP', 'Volume']
    target = 'Close'
    
    X = data[features]
    y = data[target]
    
    results = {}
    
    if model_type == "Linear Regression":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        results['rmse'] = f"{rmse:.2f}"
        
        # Create scatter plot
        fig = plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
        plt.title("Actual vs Predicted Close Values")
        plt.xlabel("Actual Close")
        plt.ylabel("Predicted Close")
        results['plot'] = fig_to_base64(fig)
        
    elif model_type == "ID3 Decision Tree":
        y_binned = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile').fit_transform(y.values.reshape(-1, 1)).ravel()
        
        X_train, X_test, y_train_binned, y_test_binned = train_test_split(X, y_binned, test_size=0.2, random_state=42)
        model = DecisionTreeClassifier(criterion='entropy')
        model.fit(X_train, y_train_binned)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test_binned, y_pred) * 100
        results['accuracy'] = f"{accuracy:.2f}%"
        
        # Create decision tree plot
        fig = plt.figure(figsize=(15, 8))
        from sklearn.tree import plot_tree
        plot_tree(model, feature_names=features, class_names=["Low", "High"], filled=True)
        results['plot'] = fig_to_base64(fig)
        
    elif model_type == "Naive Bayes":
        y_binned = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile').fit_transform(y.values.reshape(-1, 1)).ravel()
        X_train, X_test, y_train_binned, y_test_binned = train_test_split(X, y_binned, test_size=0.2, random_state=42)
        
        model = GaussianNB()
        model.fit(X_train, y_train_binned)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test_binned, y_pred) * 100
        results['accuracy'] = f"{accuracy:.2f}%"
        
        # Create confusion matrix
        cm = confusion_matrix(y_test_binned, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred Low", "Pred High"], yticklabels=["Act Low", "Act High"])
        results['plot'] = fig_to_base64(fig)
        
        # Classification report
        report = classification_report(y_test_binned, y_pred, target_names=["Low", "High"], output_dict=True)
        results['report'] = pd.DataFrame(report).T.to_html(classes='table table-striped')
    
    return render_template('analysis_results.html', 
                          model_type=model_type,
                          results=results)

@app.route('/live_ticker', methods=['GET', 'POST'])
def live_ticker():
    if request.method == 'POST':
        ticker = request.form.get('ticker')
        if not ticker:
            return render_template('live_ticker.html', error="Please enter a ticker symbol")
        
        data = load_stock_data(ticker)
        if data.empty:
            return render_template('live_ticker.html', error=f"No data available for {ticker}")
        
        data = generate_signal(data)
        last_signal = data['Signal'].iloc[-1] if len(data) > 50 else 0
        action = recommendation(last_signal)
        
        # Create charts
        data_preview = data.tail().to_html(classes='table table-striped')
        
        # Closing price chart
        fig_close = px.line(data, x='Date', y='Close', title=f"Closing Price of {ticker}")
        close_chart = create_plotly_json(fig_close)
        
        # EMA chart
        ema_period = 20  # Default EMA period
        data['EMA'] = calculate_ema(data, ema_period)
        fig_ema = px.line(data, x='Date', y=['Close', 'EMA'], title=f"EMA ({ema_period}) vs Close for {ticker}")
        ema_chart = create_plotly_json(fig_ema)
        
        # Moving Average Crossover chart
        fig_ma = px.line(data, x='Date', y=['Close', 'Short_MA', 'Long_MA'], title=f"MA Crossover for {ticker}")
        ma_chart = create_plotly_json(fig_ma)
        
        # Daily Return Distribution
        returns = data['Close'].pct_change().dropna()
        fig_ret = plt.figure()
        sns.histplot(returns, bins=50, kde=True)
        plt.title(f'Daily Return Distribution for {ticker}')
        returns_chart = fig_to_base64(fig_ret)
        
        return render_template('live_ticker.html', 
                              ticker=ticker,
                              data_preview=data_preview,
                              close_chart=close_chart,
                              ema_chart=ema_chart,
                              ma_chart=ma_chart,
                              returns_chart=returns_chart,
                              action=action,
                              signal=last_signal)
    
    return render_template('live_ticker.html')

@app.route('/compare_tickers', methods=['GET', 'POST'])
def compare_tickers():
    if request.method == 'POST':
        ticker1 = request.form.get('ticker1')
        ticker2 = request.form.get('ticker2')
        
        if not ticker1 or not ticker2:
            return render_template('compare_tickers.html', error="Please enter both ticker symbols")
        
        df1 = load_stock_data(ticker1)
        df2 = load_stock_data(ticker2)
        
        if df1.empty or df2.empty:
            return render_template('compare_tickers.html', 
                                  error=f"No data available for one or both tickers")
        
        df1 = df1.rename(columns={'Close': f'Close_{ticker1}'})
        df2 = df2.rename(columns={'Close': f'Close_{ticker2}'})
        merged = pd.merge(df1[['Date', f'Close_{ticker1}']], df2[['Date', f'Close_{ticker2}']], on='Date')
        
        # Comparison chart
        fig = px.line(merged, x='Date', y=[f'Close_{ticker1}', f'Close_{ticker2}'],
                      title=f"Stock Closing Price Comparison: {ticker1} vs {ticker2}")
        comparison_chart = create_plotly_json(fig)
        
        # Returns distribution
        returns1 = df1.set_index('Date')[f'Close_{ticker1}'].pct_change()
        returns2 = df2.set_index('Date')[f'Close_{ticker2}'].pct_change()
        
        combined_returns = pd.concat([returns1, returns2], axis=1).dropna()
        combined_returns.columns = [f'{ticker1} Return', f'{ticker2} Return']
        
        fig_ret = plt.figure()
        sns.kdeplot(combined_returns[f'{ticker1} Return'], label=ticker1)
        sns.kdeplot(combined_returns[f'{ticker2} Return'], label=ticker2)
        plt.legend()
        plt.title("Daily Returns Distribution")
        returns_chart = fig_to_base64(fig_ret)
        
        return render_template('compare_tickers.html',
                              ticker1=ticker1,
                              ticker2=ticker2,
                              comparison_chart=comparison_chart,
                              returns_chart=returns_chart)
    
    return render_template('compare_tickers.html')

@app.route('/risk_analysis', methods=['GET', 'POST'])
def risk_analysis():
    if request.method == 'POST':
        if 'files[]' not in request.files:
            return render_template('risk_analysis.html', error="No files part")
        
        files = request.files.getlist('files[]')
        if not files or files[0].filename == '':
            return render_template('risk_analysis.html', error="No selected files")
        
        stock_returns = pd.DataFrame()
        tickers = []
        
        for file in files:
            if file and file.filename.endswith('.csv'):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                try:
                    df = pd.read_csv(filepath)
                    if 'Date' not in df.columns or 'Close' not in df.columns:
                        continue
                    
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.sort_values('Date')
                    ticker = filename.replace(".csv", "")
                    tickers.append(ticker)
                    df.set_index('Date', inplace=True)
                    
                    # Calculate returns
                    df[f'{ticker}_Return'] = df['Close'].pct_change()
                    stock_returns = pd.concat([stock_returns, df[f'{ticker}_Return']], axis=1)
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        if stock_returns.empty:
            return render_template('risk_analysis.html', error="No valid data to analyze")
        
        stock_returns.dropna(inplace=True)
        returns_preview = stock_returns.tail().to_html(classes='table table-striped')
        
        # Volatility chart
        volatilities = stock_returns.std()
        fig_vol = px.bar(x=volatilities.index, y=volatilities.values, title="Volatility Comparison")
        volatility_chart = create_plotly_json(fig_vol)
        
        # Average returns chart
        avg_returns = stock_returns.mean()
        fig_avg = px.bar(x=avg_returns.index, y=avg_returns.values, title="Average Returns")
        avg_returns_chart = create_plotly_json(fig_avg)
        
        # Correlation matrix
        corr = stock_returns.corr()
        fig_corr = plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title("Correlation Matrix")
        corr_chart = fig_to_base64(fig_corr)
        
        # Recommendations
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
        recommendations = action_df.to_html(classes='table table-striped')
        
        return render_template('risk_analysis.html',
                              returns_preview=returns_preview,
                              volatility_chart=volatility_chart,
                              avg_returns_chart=avg_returns_chart,
                              corr_chart=corr_chart,
                              recommendations=recommendations)
    
    return render_template('risk_analysis.html')

@app.route('/api/update_ema', methods=['POST'])
def update_ema():
    data = request.json
    ticker = data.get('ticker')
    ema_period = int(data.get('period', 20))
    
    stock_data = load_stock_data(ticker)
    if stock_data.empty:
        return jsonify({'error': f"No data available for {ticker}"})
    
    stock_data['EMA'] = calculate_ema(stock_data, ema_period)
    
    fig_ema = px.line(stock_data, x='Date', y=['Close', 'EMA'], title=f"EMA ({ema_period}) vs Close for {ticker}")
    return jsonify({'chart': create_plotly_json(fig_ema)})

if __name__ == '__main__':
    app.run(debug=True)
