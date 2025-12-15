import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Technical Analysis library
import ta

# Load data
data = pd.read_csv("DOGE-USD.csv", parse_dates=['Date'])

# Print correlation matrix
print("Correlation Matrix:")
print(data.corr())

# Check for null values
if data.isnull().any().any():
    print("Null values found. Dropping them.")
    data.dropna(inplace=True)

# Set date as index
data.set_index('Date', inplace=True)

# Print summary statistics
print("Summary Statistics:")
print(data.describe())

# Plot close price over time
plt.figure(figsize=(20, 7))
data['Close'].plot(linewidth=2.5, color='b')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title("Date vs Close of 2021")
plt.show()


# Create new features
data["gap"] = (data["High"] - data["Low"]) * data["Volume"]
data["y"] = data["High"] / data["Volume"]
data["z"] = data["Low"] / data["Volume"]
data["a"] = data["High"] / data["Low"]
data["b"] = (data["High"] / data["Low"]) * data["Volume"]

# Technical indicators
data['SMA_10'] = ta.trend.sma_indicator(data['Close'], window=10)
data['EMA_10'] = ta.trend.ema_indicator(data['Close'], window=10)
data['RSI_14'] = ta.momentum.rsi(data['Close'], window=14)

# Lagged features
data['Close_lag1'] = data['Close'].shift(1)
data['Close_lag2'] = data['Close'].shift(2)
data['Volume_lag1'] = data['Volume'].shift(1)

# Print correlation with close price
print("Correlation with Close Price:")
print(abs(data.corr()["Close"].sort_values(ascending=False)))


# Select relevant features (including new ones)
feature_cols = [
    "Close", "Volume", "gap", "a", "b",
    "SMA_10", "EMA_10", "RSI_14",
    "Close_lag1", "Close_lag2", "Volume_lag1"
]
data = data[feature_cols]

# Print first few rows
print("Data Head:")
print(data.head())

# Split data into training and testing sets
df2 = data.tail(30)
train = df2[:11]
test = df2[-19:]

print("Train Shape:", train.shape)
print("Test Shape:", test.shape)


# Grid search for best SARIMAX (p,d,q)
import warnings
warnings.filterwarnings("ignore")
import itertools

exog_cols = [col for col in train.columns if col != "Close"]

p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
best_aic = np.inf
best_order = None
best_model = None

for order in pdq:
    try:
        model = SARIMAX(endog=train["Close"], exog=train[exog_cols], order=order)
        results = model.fit(disp=False)
        if results.aic < best_aic:
            best_aic = results.aic
            best_order = order
            best_model = results
    except Exception:
        continue

print(f"Best SARIMAX order: {best_order} (AIC={best_aic:.2f})")
print("Model Summary:")
print(best_model.summary())


# Make predictions (SARIMAX)
start = 11
end = 29
sarimax_preds = best_model.predict(start=start, end=end, exog=test[exog_cols])

# RandomForestRegressor for comparison
from sklearn.metrics import mean_squared_error
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(train[exog_cols], train["Close"])
rf_preds = rf.predict(test[exog_cols])

# Calculate RMSE for both models
sarimax_rmse = np.sqrt(mean_squared_error(test["Close"], sarimax_preds))
rf_rmse = np.sqrt(mean_squared_error(test["Close"], rf_preds))
print(f"SARIMAX RMSE: {sarimax_rmse:.4f}")
print(f"RandomForest RMSE: {rf_rmse:.4f}")

# Plot actual vs predicted close price for both models
plt.figure(figsize=(12, 6))
plt.plot(test.index, test["Close"], label='Actual', linewidth=2)
plt.plot(test.index, sarimax_preds, label='SARIMAX Predicted', linestyle='--')
plt.plot(test.index, rf_preds, label='RandomForest Predicted', linestyle=':')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title("Actual vs Predicted Close Price (SARIMAX vs RandomForest)")
plt.legend()
plt.show()