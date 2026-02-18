import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np
import math
#Create an estimated demand using moving average
data = pd.read_csv('/Users/ishaanmarwaha/venv/Samara/GAB_partsMonthly.csv')
df = pd.DataFrame(data)

sal_qty_cols = [f"Sal Qty {i:02d}" for i in range(1,13)]
df['avg_12m'] = df[sal_qty_cols].sum(axis=1) / 12
df = df[df['avg_12m'] != 0]

id_vars = ['Seq No', 'Itm Code', 'Function Description', 'Opn Qty']

# We'll collect for each month
records = []

for month in range(1, 13):
    pur_col = f'Pur Qty {month:02d}'
    sal_col = f'Sal Qty {month:02d}'
    bal_col = f'Bal Qty {month:02d}'
    for idx, row in df.iterrows():
        records.append({
            'Seq No': row['Seq No'],
            'Itm Code': row['Itm Code'],
            'Function Description': row['Function Description'],
            'Month': month,
            'Pur_Qty': row[pur_col],
            'Sal_Qty': row[sal_col],
            'Bal_Qty': row[bal_col],
            # you could add 'Opn Qty' (open qty at the beginning) and other features as needed
        })

long_df = pd.DataFrame(records)

# Optional: Sort and clean data
long_df = long_df.sort_values(['Itm Code', 'Month'])

long_df['Sal_Qty_Lag1'] = long_df.groupby('Itm Code')['Sal_Qty'].shift(1)
long_df['Sal_Qty_Roll3'] = long_df.groupby('Itm Code')['Sal_Qty'].transform(lambda x: x.rolling(3, 1).mean())



# If you want to use part IDs as a categorical feature:
le = LabelEncoder()
long_df['Part_Code_Encoded'] = le.fit_transform(long_df['Itm Code'])

# Remove first month per part (missing lag)
long_df = long_df[long_df['Sal_Qty_Lag1'].notna()]

# Features for model
feature_cols = ['Pur_Qty', 'Bal_Qty', 'Sal_Qty_Lag1', 'Sal_Qty_Roll3', 'Month', 'Part_Code_Encoded']
target_col = 'Sal_Qty'

X = long_df[feature_cols]
y = long_df[target_col]

df['counts'] = long_df.groupby('Itm Code').size()


# Split train/test: e.g., last month per part as test set
long_df['Month_Rank'] = long_df.groupby('Itm Code').cumcount()
train = long_df[long_df['Month_Rank'] < 11]
test = long_df[long_df['Month_Rank'] == 11]

X_train, X_test = train[feature_cols], test[feature_cols]
y_train, y_test = train[target_col], test[target_col]

# Model training: Random Forest (LightGBM is similar)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
from sklearn.metrics import mean_squared_error
y_pred = model.predict(X_test)
print("Test RMSE:", mean_squared_error(y_test, y_pred, squared=False))