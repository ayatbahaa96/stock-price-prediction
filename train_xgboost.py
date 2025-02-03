import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


# Fetch Stock Data
def fetch_stock_data(ticker, start='2020-01-01', end='2024-01-01'):
    df = yf.download(ticker, start=start, end=end)
    expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if 'Adj Close' in df.columns:
        expected_columns.append('Adj Close')
    df.columns = expected_columns
    df.reset_index(inplace=True)
    print(df.head())
    return df

df = fetch_stock_data('AAPL')

# Add Technical Indicators
def add_technical_indicators(df):
    rsi = ta.momentum.RSIIndicator(df['Close'])
    df['RSI'] = rsi.rsi()
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    bb = ta.volatility.BollingerBands(df['Close'])
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df.dropna(inplace=True)
    print(df[['RSI', 'MACD', 'BB_High', 'BB_Low', 'SMA_50', 'SMA_200']].head())
    return df

df = add_technical_indicators(df)

# Prepare Data for XGBoost
def prepare_data(df):
    df = df[['Close', 'RSI', 'MACD', 'BB_High', 'BB_Low', 'SMA_50', 'SMA_200']]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    X, y = [], []
    for i in range(60, len(scaled_data) - 1):
        X.append(scaled_data[i-60:i].flatten())
        y.append(scaled_data[i + 1, 0])
    X, y = np.array(X), np.array(y)
    print(f"Training data shape: {X.shape}, Labels shape: {y.shape}")
    return train_test_split(X, y, test_size=0.2, random_state=42), scaler

(X_train, X_test, y_train, y_test), scaler = prepare_data(df)

# Train XGBoost Model
model = XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.05)
model.fit(X_train, y_train)

# Make Predictions
predictions = model.predict(X_test)

# Ensure correct shape for inverse transformation
predictions = predictions.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
num_features = X_train.shape[1] // 60
zeros_placeholder = np.zeros((predictions.shape[0], num_features - 1))
predictions_full = np.hstack((predictions, zeros_placeholder))
y_test_full = np.hstack((y_test, zeros_placeholder))
predictions = scaler.inverse_transform(predictions_full)[:, 0]
y_test = scaler.inverse_transform(y_test_full)[:, 0]


# Plot results
plt.figure(figsize=(10, 5))
plt.plot(y_test, label='Real Prices', linestyle='dashed')
plt.plot(predictions, label='Predicted Prices')
plt.legend()
plt.title("Stock Price Prediction (XGBoost)")
plt.show()

# Evaluate Model
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mape = mean_absolute_percentage_error(y_test, predictions) * 100
print('Evaluation of XGBoost model')
print(f"XGBoost RMSE: {rmse}")
print(f"XGBoost MAPE: {mape}%")



import pandas as pd

# Assuming actual and predicted values are available as lists or arrays
# Replace these with actual model outputs if available

# Example date range (replace with actual dates)
dates = pd.date_range(start="2023-01-01", periods=60, freq='D')

# Example predictions and actual values (replace with actual data)
actual_values = [round(value, 2) for value in list(np.random.uniform(120, 200, size=len(dates)))]
lstm_predictions = [round(value, 2) for value in list(np.array(actual_values) + np.random.uniform(-10, 10, size=len(dates)))]
xgboost_predictions = [round(value, 2) for value in list(np.array(actual_values) + np.random.uniform(-5, 5, size=len(dates)))]

# Create DataFrame
comparison_df = pd.DataFrame({
    'Date': dates,
    'Actual Price': actual_values,
    'LSTM Predicted Price': lstm_predictions,
    'XGBoost Predicted Price': xgboost_predictions
})

# Display table for user
print(comparison_df)