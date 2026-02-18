import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error
#Create an estimated demand using moving average

excel_path = '/Users/ishaanmarwaha/venv/Samara/New Form Data/FM_FDX_MOG_data.xlsx'

sheet_names = pd.ExcelFile(excel_path).sheet_names





def process_sheet(df, brand_name):
    # Flatten columns to use only the first header row (column codes)
    df.columns = [col[0] for col in df.columns]

    # Fill down the Month column
    df['Month'] = df['Month'].ffill()

    # Only keep rows for the metrics you care about
    metrics = ['Opn Qty', 'Pur Qty', 'Sal Qty', 'Bal Qty']
    df = df[df['Itm Code'].isin(metrics)]

    # Only keep 'Opn Qty' for the first month
    first_month = df['Month'].iloc[0]
    df = df[(df['Itm Code'] != 'Opn Qty') | (df['Month'] == first_month)]

    # All part columns = everything except 'Month' and 'Itm Code'
    value_columns = [col for col in df.columns if col not in ['Month', 'Itm Code']]

    # Melt to long format
    long_df = df.melt(
        id_vars=['Month', 'Itm Code'],
        value_vars=value_columns,
        var_name='Part_Code',
        value_name='Value'
    )

    # Only keep numeric values (drop blanks/NaNs)
    long_df = long_df[pd.to_numeric(long_df['Value'], errors='coerce').notnull()]

    # Add Brand column and tidy up
    long_df['Brand'] = brand_name
    long_df = long_df.rename(columns={'Itm Code': 'Inv_data'})
    return long_df[['Month', 'Part_Code', 'Inv_data', 'Value', 'Brand']]

final_long_df = pd.DataFrame()

for sheet in sheet_names:
    df = final_long_df
    cleaned = process_sheet(df, brand_name=sheet)
    final_long_df = pd.concat([final_long_df, cleaned], ignore_index=True)
    
    pivot_df = df.pivot_table(
    index=['Month', 'Part_Code'],
    columns='Inv_data',
    values='Value',
    aggfunc='first'  # or 'sum' depending on structure
).reset_index()
    

pivot_df.columns.name = None

final_long_df.to_csv('cleaned_parts_data.csv', index=False)





    
