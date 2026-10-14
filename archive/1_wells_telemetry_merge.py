import pandas as pd
import geopandas as gpd
import numpy as np
import os

print("--- Processing GWL Data (Manual + Telemetry) ---")
# 1. Load Manual Data
w_man = pd.read_csv("manual_dataset&ground_truth/gwl_manual_quarterly_cgwb_ka_2021_2025.csv")
w_man["datetime"] = pd.to_datetime(w_man["Data Acquisition Time"], dayfirst=True)
w_man["gwl"] = w_man["Groundwater Level Quarterly Manual (meter)"]
w_man = w_man[["Station", "Latitude", "Longitude", "datetime", "gwl"]].copy()

# 2. Load Telemetry Data (2021-2025)
print("Loading telemetry data (this might take a moment)...")
try:
    w_tel = pd.read_csv("manual_dataset&ground_truth/gwl_tel_6_hourly_karnataka_ka_2021_2025.csv")
    # You will need to check the exact column names of your telemetry CSV here.
    # Assuming they are similar to the manual dataset for now. Adjust if needed.
    tel_date_col = "Data Acquisition Time" # Replace if different
    tel_gwl_col = "Groundwater Level 6 Hourly Telemetry (meter)" # Replace if different
    
    # Check actual columns
    cols = w_tel.columns.tolist()
    if tel_date_col not in cols or tel_gwl_col not in cols:
        print(f"Telemetry columns: {cols}")
        # Try to guess
        tel_date_col = next((c for c in cols if 'time' in c.lower() or 'date' in c.lower()), cols[0])
        tel_gwl_col = next((c for c in cols if 'meter' in c.lower() or 'level' in c.lower()), cols[-1])
        
    w_tel["datetime"] = pd.to_datetime(w_tel[tel_date_col], dayfirst=True, errors='coerce')
    w_tel["gwl"] = w_tel[tel_gwl_col]
    w_tel = w_tel[["Station", "Latitude", "Longitude", "datetime", "gwl"]].copy()
    w_all = pd.concat([w_man, w_tel], ignore_index=True)
    print(f"Combined manual + telemetry. Total raw records: {len(w_all)}")
except Exception as e:
    print(f"Could not load telemetry data: {e}. Proceeding with manual only.")
    w_all = w_man

# 3. Filter to Pre-Monsoon (Mar-May) & Clean Outliers
w_all["month"] = w_all["datetime"].dt.month
w_all["year"] = w_all["datetime"].dt.year
pm = w_all[w_all["month"].isin([3, 4, 5])].copy()

# Remove extreme outliers (like the 133m reading, assuming 60m is a safe max depth for this region).
pm = pm[(pm["gwl"] >= 0) & (pm["gwl"] < 60)]

# 4. Aggregate
agg = pm.groupby(["Station", "year"]).agg(
    gwl=("gwl", "mean"), lat=("Latitude", "first"), lon=("Longitude", "first")
).reset_index().rename(columns={"Station": "well_id"})

# 5. Spatial Join
gdf = gpd.GeoDataFrame(agg, geometry=gpd.points_from_xy(agg["lon"], agg["lat"]), crs="EPSG:4326").to_crs("EPSG:32643")
taluk = gpd.read_file("labeled_taluks.gpkg").to_crs("EPSG:32643")
gdf = gpd.sjoin(gdf, taluk[["key", "label_binary", "geometry"]], how="inner", predicate="within").rename(columns={"key": "taluk_id"})
gdf["x"], gdf["y"] = gdf.geometry.x, gdf.geometry.y

# 6. Local Stressed Label
taluk_stats = gdf.groupby("taluk_id")["gwl"].agg(["mean", "std"]).rename(columns={"mean": "t_mean", "std": "t_std"})
gdf = gdf.merge(taluk_stats, left_on="taluk_id", right_index=True)
gdf["gwl_zscore"] = (gdf["gwl"] - gdf["t_mean"]) / gdf["t_std"].clip(lower=0.01)
gdf["stressed"] = (gdf["gwl_zscore"] > 1.0).astype(float) # Float for PyTorch

out_cols = ["well_id", "x", "y", "year", "gwl", "stressed", "taluk_id", "label_binary"]
gdf[out_cols].to_parquet("wells_premonsoon.parquet", index=False)
print(f"Saved wells_premonsoon.parquet. Total well-years inside 54 taluks: {len(gdf)}")
