import pandas as pd


df = pd.read_csv('/Users/ishaanmarwaha/venv/Samara/FDX_partsMonthly - Sheet1.csv',
                skiprows=2,            # throw away “SAMARA AUTO SUPPLIES LTD.…”
    header=[0,1],          # use the next two lines as header levels
    quotechar='"',         # so that SUPPORT, UPPER RAD stays one field
    skipinitialspace=True
)

print(df.columns)
flat_cols = []
for lvl0,lvl1 in df.columns:
    lvl0, lvl1 = lvl0.strip(), lvl1.strip()
    if lvl0 == "Month":
        flat_cols.append("Month")
    elif lvl0 == "Itm Code" and lvl1 == "Function Description":
        flat_cols.append("Metric")
    else:
        flat_cols.append(f"{lvl0} — {lvl1}")
df.columns = flat_cols

long = df.melt(
    id_vars=["Month","Metric"],
    var_name="Part",
    value_name="Value"
)
long[["Part_Code","Function_Description"]] = (
    long["Part"]
     .str.split(" — ", n=1, expand=True)
)
long = long.drop(columns="Part")

# (Optionally) 5) If you want the final “flat” shape with Sal_Qty / Pur_Qty / Bal_Qty 
#    as separate columns again, just pivot:
flat = (
    long
     .pivot_table(
        index=["Part_Code","Function_Description","Month"],
        columns="Metric",
        values="Value",
        aggfunc="first"
     )
     .reset_index()
     .rename_axis(columns=None)
)

flat["Seq_No"] = flat["Part_Code"].factorize()[0] + 1

# 7) Re‐order to match your example:
flat = flat[
    ["Part_Code","Month","Function_Description","Seq_No",
     "Opn Qty","Pur Qty","Sal Qty","Bal Qty"]
]



df.to_csv('/Users/ishaanmarwaha/venv/Samara/FDX_partsMonthly-reformat', index = False)