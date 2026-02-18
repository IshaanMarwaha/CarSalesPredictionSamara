from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


df = pd.read_csv('Samara/CarParts_Testing - Sheet1.csv')


#SEPARATING DATE
df = pd.DataFrame(data)
df.to_csv('car_parts_60_months.csv', index=False)



#WEIGHTED MONTHLY
n = len(df)

group_number = n // 4 if n >= 4 else 1
group_weights = [0.55, 0.85, 1.15, 1.45]
weights = np.repeat(group_weights, group_number)
if len(weights) < n:  # Handle extra months (if n not divisible by 4)
    weights = np.concatenate([weights, np.full(n - len(weights), group_weights[-1])])

df['Weighted_Sales'] = df['Sales'] * weights

#ESTIMATED DEMAND
inc = 0
total_demand = 0
while inc < n:
    total_demand += df.at[inc, 'Weighted_Sales']
    inc += 1
    
df['Estimated_Demand'] = total_demand / n

#UNMATCHED DEMAND
"""
unmatch_demand = 0
for stock in df['Curr_Stock']:
    if stock == 0 or stock <= df['Sales']:
        unmatch_demand += 1
"""
        




# Estimate stock for next month
df['Predicted_Stock_1'] = df['Curr_Stock'] - df['Sales'] + df['Estimated_Demand']

# Lag features
df['Sales_Lag_1'] = df['Sales'].shift(1)
df['Stock_Lag_1'] = df['Curr_Stock'].shift(1)


df.dropna(inplace=True)

features = ['Sales_Lag_1', 'Stock_Lag_1', 'Estimated_Demand']
target = 'Curr_Stock'  # or the stock of the next month

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

model = RandomForestRegressor()
model.fit(X_train, y_train)

df['Predicted_Stock_2'] = model.predict(X)
df['Predicted_Stock'] = ((df['Predicted_Stock_1'] + (df['Predicted_Stock_2'] * 4)) / 5)



print(df['Predicted_Stock'],df['Estimated_Demand'])

