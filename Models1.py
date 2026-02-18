from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math

df = pd.read_csv('cleaned_parts_data.csv')
df = pd.DataFrame(df)

month_order = df['Month'].drop_duplicates()

# Step 2: set Month as a categorical with that order
df['Month'] = pd.Categorical(df['Month'], categories=month_order, ordered=True)
df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')


df = df.pivot_table(
    index=['Part_Code', 'Month'],
    columns=['Inv_data'],
    values='Value',
    aggfunc='first'  # or 'sum' depending on structure
).reset_index()

df['Month_Order'] = df.groupby('Part_Code').cumcount()


df.columns.name = None 

#set Opn Qty to first Balance month
df = df.sort_values(by=['Part_Code', 'Month'])
df['Bal_Qty_Set'] = df.groupby('Part_Code')['Bal Qty'].shift(1)
first_month_idx = df.groupby('Part_Code')['Month'].idxmin()
df.loc[first_month_idx, 'Bal_Qty_Set'] = df.loc[first_month_idx, 'Opn Qty']
df['Bal_Qty_Set'] = df['Bal_Qty_Set'].astype(int)


df['Yearly_Avg'] = df.groupby('Part_Code')['Sal Qty'].transform('mean')

#drop all parts without sales
no_yearly = df['Yearly_Avg'] == 0
df = df[~no_yearly]

#Number of unique parts
num_parts = len(df['Part_Code'].unique())
print('Number of Parts: ', num_parts)

#Replace stockout items
df['Stockout'] = df['Bal_Qty_Set'] < df['Yearly_Avg']

#Create 3 month avg for non-stockout months
df_no_stockout = df[df['Stockout'] == False].copy()
df_no_stockout['Rolling3_Avg'] = (
     df_no_stockout
      .groupby('Part_Code')['Sal Qty']     # or the column you want to average
      .transform(lambda x: x.rolling(3, min_periods=1).mean())
)


df = df.merge(
    df_no_stockout[['Part_Code', 'Month', 'Rolling3_Avg']],
    on=['Part_Code', 'Month'],
    how='left'
)

df.to_csv('MLlongSetData', index = False)

conditions = [
    df['Month_Order'].isin([1, 2, 3]),
    df['Month_Order'].isin([4, 5, 6]),
    df['Month_Order'].isin([7, 8, 9]),
    df['Month_Order'].isin([10, 11, 12]),
]

weights = [0.55, 0.85, 1.15, 1.45]

# Apply using np.select
df['Quarter_Weight'] = np.select(conditions, weights, default=np.nan)

# Now you can compute Weighted Sales
df['Weighted_Sales'] = np.round(df['Rolling3_Avg'] * df['Quarter_Weight'])

"""
def correct_sales(group):
    sales = group['Sal Qty'].astype(float).copy()
    stock = group['Adj_Bal_Qty'].astype(float)
    
    for i in range(len(sales)):
        if df['Stockout']:
"""          
#Adding more features
lead_time = 4
df['Minimum_Inv'] = round(lead_time * df['Yearly_Avg'])# made both measures worse
df['Lag_1_Sales'] = df.groupby('Part_Code')['Sal Qty'].shift(1)# made both measures worse
df['Prev_Stockout'] = df.groupby('Part_Code')['Stockout'].shift(1)#made both measures worse
df['Sales_Change'] = df.groupby('Part_Code')['Sal Qty'].pct_change() #increased correlation but worsened predictiveness
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df['Stockout_3_Months'] = (
    df.groupby('Part_Code')['Stockout']
      .transform(lambda x: x.rolling(3, min_periods=1).mean())
)

df['Lag_1_Sales'] = df.groupby('Part_Code')['Sal Qty'].shift(1).fillna(0)
df['Rolling3_Avg'] = df['Rolling3_Avg'].fillna(df['Yearly_Avg']) # or another logical fill

df['Expanding_Avg'] = (
    df.groupby('Part_Code')['Sal Qty']
      .expanding()
      .mean()
      .reset_index(level=0, drop=True)
)
df['Target_Next_Sales'] = df.groupby('Part_Code')['Sal Qty'].shift(-1)

df_model = df[df['Target_Next_Sales'].notna()].copy()

last_idx = df.groupby('Part_Code')['Month_Order'].idxmax()



test_idx = df_model.groupby('Part_Code')['Month_Order'].idxmax()
train_df = df_model.drop(index=test_idx)
test_df  = df_model.loc[test_idx]

features = [
    'Pur Qty' , 'Month_Order' , 'Bal_Qty_Set', 'Rolling3_Avg', 'Weighted_Sales','Expanding_Avg','Sal Qty'
]


X_train = train_df[features]
y_train = train_df['Target_Next_Sales'].loc[X_train.index]

X_test = test_df[features]
y_test = test_df['Target_Next_Sales'].loc[X_test.index]
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)


df['Predicted_Next_Month_Sales'] = np.nan


df.loc[X_test.index, 'Predicted_Next_Month_Sales'] = np.round(y_pred_test)


last_rows = df.loc[last_idx, features].fillna(0)
df.loc[last_idx, 'Predicted_Next_Month_Sales'] = np.round(model.predict(last_rows))


print(df[df['Month_Order'] == 12][features + ['Sal Qty']])


notna_mask = ~y_test.isna()
y_test_filtered = y_test[notna_mask]
y_pred_filtered = y_pred_test[notna_mask]

# Now safe to compute metrics


comparison = pd.DataFrame({
    'Actual_Sales': y_test,
    'Predicted_Sales': np.round(y_pred_test)
})

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test_filtered, y_pred_filtered)


print(f"Mean Absolute Error (MAE): {mae:.2f}")

from sklearn.metrics import r2_score
r2 = r2_score(y_test_filtered, y_pred_filtered)

print(f"R² Score: {r2:.2f}")

comparison = pd.DataFrame({
    'Actual_Sales': y_test,
    'Predicted_Sales': y_pred_test
})
comparison['Pct_Error'] = 100 * np.abs(comparison['Predicted_Sales'] - comparison['Actual_Sales']) / comparison['Actual_Sales'].replace(0, np.nan)
mape = comparison['Pct_Error'].mean()
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")



on_hand = df['Bal Qty']  # or Bal Qty
sigma = df['Sal Qty']
pred_monthly = df['Predicted_Next_Month_Sales']



target_level = pred_monthly * 4
order_qty = target_level - on_hand

dfModel = df[['Part_Code','Month_Order','Month', 'Stockout','Bal Qty', 'Sal Qty','Predicted_Next_Month_Sales','Target_Next_Sales']]
dfModel.to_csv("scored_parts.csv", index=False)

target_month = pd.Timestamp("2025-06-01")
dfExcel = df[['Part_Code','Month', 'Stockout','Bal Qty', 'Sal Qty','Predicted_Next_Month_Sales']]
dfExcel = dfExcel.rename(columns={'Predicted_Next_Month_Sales': 'Estimated Demand'})
dfJune = dfExcel[dfExcel['Month'] == target_month]
dfExcel.to_csv("Models1data.csv", index = False)
dfJune.to_csv("Model1Junedata.csv", index = False)

