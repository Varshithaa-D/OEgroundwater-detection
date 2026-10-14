import pandas as pd
try:
    w = pd.read_csv("manual_dataset&ground_truth/gwl_manual_quarterly_cgwb_ka_2021_2025.csv")
    w["t"] = pd.to_datetime(w["Data Acquisition Time"], dayfirst=True, errors="coerce")
    print(w.groupby([w.t.dt.year, w.t.dt.month]).size().unstack(fill_value=0))
except Exception as e:
    print("Error reading CSV:", e)
