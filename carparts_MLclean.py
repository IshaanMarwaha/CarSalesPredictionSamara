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


"""
wide_df = final_long_df.pivot_table(
    index=['Month', 'Part_Code', 'Brand'],
    columns='Inv_data',
    values='Value'
).reset_index()

def impute_stockout_sales(group):
    sales = group['Sal Qty'].astype(float).copy()
    prev_stock = group['Bal Qty'].astype(float).shift(1)  # previous month's balance
    for i in range(len(sales)):
        # check prev_stock, not current!
        if prev_stock.iloc[i] == 0 and sales.iloc[i] == 0:
            # look back up to 6 previous periods with positive stock
            prev_sales = sales.iloc[max(0, i-6):i]
            prev_stock_hist = prev_stock.iloc[max(0, i-6):i]
            valid = prev_sales[prev_stock_hist > 0]
            if len(valid):
                sales.iloc[i] = valid.mean()
            else:
                valid_all = sales[prev_stock > 0]
                sales.iloc[i] = valid_all.mean() if len(valid_all) else 0
    return pd.Series(sales.values, index=group.index)

wide_df['Imputed_Sal_Qty'] = (
    wide_df.groupby('Part_Code', group_keys=False)
        .apply(impute_stockout_sales)
)

# 3. Optional: Merge back to long form if you want
imputed_long = pd.merge(
    final_long_df,
    wide_df[['Month', 'Part_Code', 'Brand', 'Imputed_Sal_Qty']],
    on=['Month', 'Part_Code', 'Brand'],
    how='left'
)

imputed_long.loc[imputed_long['Inv_data'] == 'Sal Qty', 'Value'] = (
    imputed_long.loc[imputed_long['Inv_data'] == 'Sal Qty', 'Imputed_Sal_Qty']
)
imputed_long = imputed_long.drop(columns=['Imputed_Sal_Qty'])
imputed_long.to_csv('cleaned_parts_data.csv', index=False)



"""
month_order = df['Month'].drop_duplicates()

# Step 2: set Month as a categorical with that order
df['Month'] = pd.Categorical(df['Month'], categories=month_order, ordered=True)
pivot_df = df.pivot_table(
    index=['Part_Code', 'Month'],
    columns=['Inv_data'],
    values='Value',
    aggfunc='first'  # or 'sum' depending on structure
).reset_index()

pivot_df.columns.name = None 

pivot_df = pivot_df.sort_values(by=['Part_Code', 'Month'])
pivot_df['Adj_Bal_Qty'] = pivot_df.groupby('Part_Code')['Bal Qty'].shift(1)


month_map = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
    'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
    'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

# Extract the first 3 letters of each Month entry and map to number
pivot_df['Month_Num'] = pivot_df['Month'].str[:3].map(month_map)



def correct_sales(group):
    sales = group['Sal Qty'].astype(float).copy()
    stock = group['Adj_Bal_Qty'].astype(float)

    for i in range(len(sales)):
        if stock.iloc[i] == 0 and sales.iloc[i] == 0:
            prev_sales = sales.iloc[max(0, i-6):i][stock.iloc[max(0, i-6):i] > 0]
            if len(prev_sales):
                sales.iloc[i] = prev_sales.mean()
    return sales

pivot_df['Corrected_Sal_Qty'] = (
    np.round(pivot_df.groupby('Part_Code').apply(correct_sales).reset_index(level=0, drop=True))
)

pivot_df['Target_Next_Sales'] = (
    pivot_df.groupby('Part_Code')['Corrected_Sal_Qty'].shift(-1)
)

conditions = [
    pivot_df['Month_Num'].isin([1, 2, 3]),
    pivot_df['Month_Num'].isin([4, 5, 6]),
    pivot_df['Month_Num'].isin([7, 8, 9]),
    pivot_df['Month_Num'].isin([10, 11, 12]),
]

# Define corresponding weights
weights = [0.55, 0.85, 1.15, 1.45]

# Apply using np.select
pivot_df['Quarter_Weight'] = np.select(conditions, weights, default=np.nan)

# Now you can compute Weighted Sales
pivot_df['Weighted_Sales'] = np.round(pivot_df['Corrected_Sal_Qty'] * pivot_df['Quarter_Weight'])
pivot_df['Avg_Weighted_Sales'] = np.round(pivot_df.groupby('Part_Code')['Weighted_Sales'].transform('mean'))

pivot_df['Adj_Bal_Qty'] = np.where(
    pivot_df['Adj_Bal_Qty'] < pivot_df['Avg_Weighted_Sales'],
    pivot_df['Avg_Weighted_Sales'],
    pivot_df['Adj_Bal_Qty']
)


features = [
    'Pur Qty', 'Adj_Bal_Qty','Month_Num','Weighted_Sales', 'Avg_Weighted_Sales'
]

full = pivot_df[features + ['Target_Next_Sales']].dropna()

X = full[features]
y = full['Target_Next_Sales']
# dropna to avoid incomplete rows

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestRegressor()
model.fit(X_train, y_train)

predictions = np.round(model.predict(X))
pivot_df.loc[X.index, 'Predicted_Next_Month_Sales'] = np.round(predictions)
pivot_df['Average Predicted'] = np.round(pivot_df.groupby('Part_Code')['Predicted_Next_Month_Sales'].transform('mean'))



pivot_df['Base Demand'] = ((pivot_df['Average Predicted'] + pivot_df['Avg_Weighted_Sales']) / 2) 
pivot_df['Monthly Stock Necessary'] = np.where(
    (pivot_df['Base Demand'] * 4.5 - pivot_df['Adj_Bal_Qty']) < 0,
    0,
    (pivot_df['Base Demand'] * 4.5 - pivot_df['Adj_Bal_Qty'])
)

pivot_df = pivot_df[pivot_df['Month'] == 'Jun-25']
   # example
pivot_df = pivot_df[['Part_Code', 'Month', 'Pur Qty', 'Adj_Bal_Qty', 'Corrected_Sal_Qty','Base Demand', 'Monthly Stock Necessary']]

from openpyxl import load_workbook

excel_path = 'New_Predictions_FM_FDX_MOG.xlsx'
pivot_df.to_excel(excel_path, index=False, engine='openpyxl')

wb = load_workbook(excel_path)
ws = wb.active

# --- Auto-fit column widths ---
for column in ws.columns:
    max_length = 0
    column_letter = column[0].column_letter
    for cell in column:
        try:
            if cell.value:
                max_length = max(max_length, len(str(cell.value)))
        except:
            pass
    adjusted_width = max_length + 2
    ws.column_dimensions[column_letter].width = adjusted_width
wb.save(excel_path)

pivot_df.to_excel('New_Predictions_FM_FDX_MOG.xlsx', index = False)

#testing for feature importance:




