import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

def fetch_stock_data(ticker, start='2020-01-01', end='2024-01-01'):
    df = yf.download(ticker, start=start, end=end)
    return df

def add_technical_indicators(df):
    """ Add technical indicators and ensure they return single Series (not DataFrames) """

    # Relative Strength Index (RSI)
    rsi = ta.momentum.RSIIndicator(df['Close'])
    df['RSI'] = rsi.rsi()

    # Moving Average Convergence Divergence (MACD)
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()  # Extract the MACD line only

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Close'])
    df['BB_High'] = bb.bollinger_hband()  # Upper Bollinger Band
    df['BB_Low'] = bb.bollinger_lband()   # Lower Bollinger Band

    # Simple Moving Averages (SMA)
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    # Ensure no NaN values exist
    df.dropna(inplace=True)

    # Display first few rows to verify correctness
    print(df[['RSI', 'MACD', 'BB_High', 'BB_Low', 'SMA_50', 'SMA_200']].head())

    return df


df = fetch_stock_data('AAPL')  # 
df = add_technical_indicators(df)  #


import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


def fetch_stock_data(ticker, start='2020-01-01', end='2024-01-01'):
    df = yf.download(ticker, start=start, end=end)
    return df
    
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
df = fetch_stock_data('AAPL')
df = add_technical_indicators(df)