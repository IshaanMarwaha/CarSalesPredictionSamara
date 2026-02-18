import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Inventory Search", layout="wide")
st.title("Parts Inventory Search + Lead Time Reorder")

def load_data(path="scored_parts.csv"):
    df = pd.read_csv(path)
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    return df

df = load_data("scored_parts.csv")

latest_idx = df.groupby("Part_Code")["Month"].idxmax()
snap = df.loc[latest_idx].copy()

sales_std = df.groupby("Part_Code")["Sal Qty"].std(ddof=0).fillna(0)
snap = snap.merge(sales_std.rename("Sales_Std"), on="Part_Code", how="left")


st.sidebar.header("Inputs")

lead_days = st.sidebar.number_input("Lead time (days)", min_value=0.0, value=14.0, step=1.0)

service_level = st.sidebar.selectbox("Service level", ["90%", "95%", "98%"])
z = {"90%": 1.28, "95%": 1.65, "98%": 2.05}[service_level]

target_months = st.sidebar.number_input("Target coverage (months)", min_value=0.5, value=2.0, step=0.5)

st.subheader("Search")

query = st.text_input("Search by Part_Code (partial match allowed)")
filtered = snap[snap["Part_Code"].astype(str).str.contains(query, case=False, na=False)] if query else snap

st.write("Matches (top 50):")
st.dataframe(
    filtered.head(50)[["Part_Code", "Month", "Bal Qty", "Predicted_Next_Month_Sales", "Stockout"]],
    use_container_width=True
)

part = st.selectbox(
    "Select a Part_Code",
    filtered["Part_Code"].astype(str).head(500).tolist() if not filtered.empty else []
)


def recommend(row, lead_days, z, target_months):
    L = lead_days / 30.0  # months

    pred_monthly = float(row.get("Predicted_Next_Month_Sales", np.nan))
    if np.isnan(pred_monthly):
        return None

    on_hand = float(row.get("Bal Qty", 0))  # or Bal Qty
    sigma = float(row.get("Sales_Std", 0))

    demand_lt = pred_monthly * L
    safety_stock = z * sigma * np.sqrt(max(L, 1e-9))
    rop = demand_lt + safety_stock

    target_level = pred_monthly * target_months + safety_stock
    order_qty = max(0.0, target_level - on_hand)

    return {
        "Predicted monthly demand": pred_monthly,
        "Lead time (months)": L,
        "Demand during lead time": demand_lt,
        "Safety stock": safety_stock,
        "Reorder point (ROP)": rop,
        "On hand": on_hand,
        "Target level": target_level,
        "Suggested order qty": int(np.ceil(order_qty)),
    }
st.subheader("Recommendation")

if part:
    row = snap.loc[snap["Part_Code"].astype(str) == part].iloc[0]

    if "Stockout" in row and bool(row["Stockout"]):
        st.warning("This part is currently flagged as a stockout. Observed sales may be constrained by inventory.")

    rec = recommend(row, lead_days=lead_days, z=z, target_months=target_months)

    if rec is None:
        st.error("No prediction available for this part.")
    else:
        st.json({k: round(v, 2) if isinstance(v, float) else v for k, v in rec.items()})
    
    