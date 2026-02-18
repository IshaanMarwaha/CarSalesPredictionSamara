import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Load and sort the data
df = pd.read_csv('Samara/CarParts_Testing - Sheet1.csv')
df['Date'] = pd.to_datetime(df['Date'])  # Ensure 'Date' is datetime
df = df.sort_values('Date').reset_index(drop=True)

# Dynamic weight application: replicate weights if needed
weights = np.tile([0.7, 0.9, 1.1, 1.3], 3)[:len(df)]  # Extend safely to match length
df['Weighted_Sales'] = df['Sales'] * weights

# Use only past sales for rolling average (to avoid lookahead bias)
df['Estimated_Demand'] = df['Weighted_Sales'].shift(1).rolling(window=3, min_periods=1).mean()

# Drop the first row(s) with NaN due to shift/rolling
df.dropna(inplace=True)

# Feature and target selection (no Curr_Stock in features)
features = df[['Sales', 'Estimated_Demand']]
target = df['Curr_Stock']

# Train the model
model = LinearRegression()
model.fit(features, target)

# Predict and append results
df['Predicted_Stock'] = model.predict(features)

# View result
print(df[['Date', 'Sales', 'Curr_Stock', 'Estimated_Demand', 'Predicted_Stock']].head())
