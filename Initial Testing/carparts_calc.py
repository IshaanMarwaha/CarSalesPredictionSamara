import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error
#Create an estimated demand using moving average

data = pd.read_csv('/Users/ishaanmarwaha/venv/Samara/GAB_partsMonthly.csv')
data1 = pd.DataFrame(data)
# 1) First, clean your column names so they’re valid stubnames:
data1.columns = data1.columns.str.replace(' ', '_')  
data1['orig_idx'] = data1.index
# 2) Reshape:
df = pd.wide_to_long(
    data1,
    stubnames=['Sal_Qty', 'Pur_Qty', 'Bal_Qty'],
    i=['orig_idx','Itm_Code'],
    j='Month',
    sep='_',
    suffix=r'\d+'
).reset_index()

df = df.sort_values('orig_idx').drop(columns='orig_idx')

df['Month'] = df['Month'].astype(int)
df = df.sort_values(['Itm_Code','Month'])


df.drop('Opn_Qty', axis = 1, inplace = True)

df.to_csv('GAB_partsCleanData.csv',index = False)

#calculate average sales per row

df['was_stockout'] = (
  df.groupby('Itm_Code')['Bal_Qty']
    .shift(1)    # that part’s prior‐month balance
    .eq(0)       # was it zero?
  &
  df['Sal_Qty'].eq(0)
)
corrected = df['Sal_Qty'].copy()
mask      = df['was_stockout']
corrected[mask] = df.loc[~mask, 'Sal_Qty'].rolling(window=3, min_periods=1).mean()
df['Sal_Corrected'] = corrected


demand_stats = (
    df
    .groupby('Itm_Code')['Sal_Corrected']
    .agg(mu='mean', sigma='std')
    .fillna(0)
)

z = 1.645   # for ~95% service
LT = 1      # lead time in months

demand_stats['safety_stock'] = z * demand_stats['sigma'] * np.sqrt(LT)

demand_stats['Monthly Stock Necessary'] = round(demand_stats['mu'] + demand_stats['safety_stock'])

df = df.merge(
    demand_stats[['Monthly Stock Necessary']],
    left_on='Itm_Code', right_index=True
)

# assume you have a column 'On_Hand' for current stock
df['Months Available'] = round(df['Bal_Qty'] / df['Monthly Stock Necessary'])  # or your actual “available” field

# flag when it's time to order
df['to_reorder'] = df['Bal_Qty'] <= (df['Monthly Stock Necessary'] * 6)

def recommend_reorder(df:pd.DataFrame, lookback_months:int, lookahead_months:int):
    current_month = df['Month'].max()
    
    # Identify parts with at least one sale in the lookback window
    start_month = current_month - lookback_months + 1
    mask = df['Month'].between(start_month, current_month)
    recent_sales = df.loc[mask].groupby('Itm_Code')['Sal_Qty'].sum()
    eligible_parts = recent_sales[recent_sales > 0].index
    
    # Slice the "current month" records for those parts
    curr = df[(df['Month'] == current_month) & df['Itm_Code'].isin(eligible_parts)].copy()
    # Compute how much to purchase: 
    # (Monthly Stock Necessary * lookahead) - (ending balance + purchased this month)
    curr['To_Purchase'] = (
        curr['Monthly Stock Necessary'] * lookahead_months
        - (curr['Bal_Qty'] + curr['Pur_Qty'])
    ).clip(lower=0)  # never negative
    
    # Rename for clarity
    curr = curr.rename(columns={'Bal_Qty': 'On_Hand'})
    curr['Current_Month'] = current_month
    
    return curr[['Itm_Code', 'Current_Month', 'Monthly Stock Necessary', 'On_Hand', 'To_Purchase']]

lookback = 6    # e.g. only include parts that sold in the last 6 months
lookahead = 3   # e.g. plan stock for the next 3 months

# 2) Call the helper to get your reorder plan:
df_1 = recommend_reorder(df, lookback_months=lookback, lookahead_months=lookahead)

df_1 = df_1[df_1['Monthly Stock Necessary']!= 0]
df_1 = df_1[df_1['Current_Month'] == 12]



df_1.to_csv('GAB_monthlyInv.csv', index= False)




"""
for m in range(2, 13):
    prev_bal_col = df.get(df['Month'] == m -1, df['Bal_Qty'])
    pur_col      = df.get(df['Month'] == m, df['Pur_Qty'])
    sal_col      = df.get(df['Month'] == m, df['Sal_Qty'])
    # wherever the prior-month balance is 0, set that sale to NaN (i.e. “drop” the data)
    if((pur_col + prev_bal_col) == 0): 
        
    

print(df['stockout'])

df.loc[df['stockout'], 'Sales_imputed'] = (
    df.loc[~df['stockout'], 'Sales']
    .rolling(window=3, min_periods=1).mean()
)

# 1b) Make sure your item codes are strings with no extra spaces
df['Itm Code'] = df['Itm Code'].astype(str).str.strip()

sal_cols = [f"Sal Qty {i:02d}" for i in range(1,13)]
weights  = np.repeat([0.55,0.85,1.15,1.45], 3)
df['forecast_12m'] = df[sal_cols].dot(weights) / 12

def part_metric(part_id, metric_type, request):
    # Map type to prefix used in column names
    type_map = {
        'sales': 'Sal Qty',
        'purchase': 'Pur Qty',
        'balance': 'Bal Qty'
    }
    # Ensure valid type
    
    part_row = df[df['Itm Code'] == part_id]
    
    prefix = type_map[metric_type]
    cols = [f"{prefix} {i:02d}" for i in range(1, 13)]

    values = part_row[cols].sum(axis=1).values[0]
    if request == 'sum':
        return values
    elif request == 'average':
        return values / 12
    

#calculate average sales per row
df['avg_12m'] = df[sal_cols].sum(axis=1) / 12
df = df[df['avg_12m'] != 0]


df.set_index('Itm Code', inplace=True)

    
df['Estimated_Demand'] = round(df[sal_cols].dot(weights) / len(sal_cols))

df.to_excel('/Users/ishaanmarwaha/venv/Samara/GAB_partsMonthly.xlsx', index=False)
#df.to_excel
    
"""








