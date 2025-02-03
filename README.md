# stock-price-prediction
Stock Price Prediction Using LSTM &amp; XGBoost
Project Overview

This project aims to develop a predictive model for forecasting the next day's closing price of a stock using historical financial data. Two models were implemented:

LSTM (Long Short-Term Memory) - A deep learning model effective for sequential time-series data.

XGBoost (Extreme Gradient Boosting) - A machine learning model known for its efficiency in structured datasets.

This repository includes the full implementation of data preprocessing, feature engineering, model training, evaluation, and comparison between the two models.

Dataset

Data Source

Data was obtained from Yahoo Finance using the yfinance Python library.

The selected stock for this study is Apple Inc. (AAPL).

Data spans from January 1, 2020, to January 1, 2024.

Features Used

The dataset includes the following financial indicators:

Open Price

High Price

Low Price

Close Price (Target variable)

Volume

Additionally, technical indicators were incorporated:

Relative Strength Index (RSI)

Moving Average Convergence Divergence (MACD)

Bollinger Bands (BB_High, BB_Low)

Simple Moving Averages (SMA_50, SMA_200)
