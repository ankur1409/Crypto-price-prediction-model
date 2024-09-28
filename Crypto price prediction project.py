import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

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

# Print correlation with close price
print("Correlation with Close Price:")
print(abs(data.corr()["Close"].sort_values(ascending=False)))

# Select relevant features
data = data[["Close", "Volume", "gap", "a", "b"]]

# Print first few rows
print("Data Head:")
print(data.head())

# Split data into training and testing sets
df2 = data.tail(30)
train = df2[:11]
test = df2[-19:]

print("Train Shape:", train.shape)
print("Test Shape:", test.shape)

# Create and fit SARIMAX model
model = SARIMAX(endog=train["Close"], exog=train[["Volume", "gap", "a", "b"]], order=(2, 1, 1))
results = model.fit()

# Print model summary
print("Model Summary:")
print(results.summary())

# Make predictions
start = 11
end = 29
predictions = results.predict(start=start, end=end, exog=test.drop("Close", axis=1))

# Plot actual vs predicted close price
plt.figure(figsize=(12, 6))
plt.plot(test.index, test["Close"], label='Actual')
plt.plot(test.index, predictions, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title("Actual vs Predicted Close Price")
plt.legend()
plt.show()