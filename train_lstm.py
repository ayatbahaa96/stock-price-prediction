# Ensure necessary imports
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Fetch stock data
def fetch_stock_data(ticker, start='2020-01-01', end='2024-01-01'):
    df = yf.download(ticker, start=start, end=end)
    return df

# Add technical indicators
def add_technical_indicators(df):
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['BB_High'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
    df['BB_Low'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df.dropna(inplace=True)  # Remove NaN values after indicator computation
    return df

# Prepare data
def prepare_data(df):
    df = df[['Close', 'RSI', 'MACD', 'BB_High', 'BB_Low', 'SMA_50', 'SMA_200']]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    for i in range(60, len(scaled_data) - 1):  # Use past 60 days to predict next day
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i + 1, 0])

    X, y = np.array(X), np.array(y)
    return train_test_split(X, y, test_size=0.2, random_state=42), scaler

# Build LSTM Model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Fetch, preprocess, and prepare data
df = fetch_stock_data('AAPL')
df = add_technical_indicators(df)
(X_train, X_test, y_train, y_test), scaler = prepare_data(df)

# Build and train model
model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# **✅ Ensure Predictions Are Generated Before Reshaping**
predictions = model.predict(X_test)  # ✅ Fix: Generate predictions before reshaping

# Reshape predictions for inverse transformation
predictions = predictions.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Correct inverse transformation
num_features = X_train.shape[2]  # Number of features used in training (7)
zeros_placeholder = np.zeros((predictions.shape[0], num_features - 1))

# Concatenate predictions with zeros to match original feature count
predictions_full = np.hstack((predictions, zeros_placeholder))
y_test_full = np.hstack((y_test, zeros_placeholder))

# Apply inverse transformation
predictions = scaler.inverse_transform(predictions_full)[:, 0]
y_test = scaler.inverse_transform(y_test_full)[:, 0]

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(y_test, label='Real Prices', linestyle='dashed')
plt.plot(predictions, label='Predicted Prices')
plt.legend()
plt.title("Stock Price Prediction (LSTM)")
plt.show()

# Model Evaluation
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mape = mean_absolute_percentage_error(y_test, predictions) * 100

print(f"LSTM RMSE: {rmse}")
print(f"LSTM MAPE: {mape}%")
