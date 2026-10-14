"""
Phase 1 — Step 1: Build pre-monsoon well labels
Produces: wells_premonsoon.parquet
Depends on: labeled_taluks.gpkg (from Phase 0)

Run BEFORE build_zarr.py and build_index.py.
"""
import pandas as pd
import geopandas as gpd
import numpy as np

# ── Exact column names from Phase 0 audit ──
DATE_COL  = "Data Acquisition Time"
LEVEL_COL = "Groundwater Level Quarterly Manual (meter)"
WID_COL   = "Station"
LAT_COL   = "Latitude"
LON_COL   = "Longitude"

# ── 1. Load and parse ──
print("Loading GWL CSV...")
w = pd.read_csv("manual_dataset&ground_truth/gwl_manual_quarterly_cgwb_ka_2021_2025.csv")
print(f"  Total records: {len(w)}")

# Date format is DD-MM-YYYY HH:MM  (e.g. "31-08-2021 06:00")
w["datetime"] = pd.to_datetime(w[DATE_COL], dayfirst=True)
w["month"] = w["datetime"].dt.month
w["year"]  = w["datetime"].dt.year

# ── 2. Filter to pre-monsoon: March, April, May ──
pm = w[w["month"].isin([3, 4, 5])].copy()
print(f"  Pre-monsoon (Mar-May) records: {len(pm)}")

# If very few Mar-May records, also include January (CGWB's "pre-monsoon winter")
if len(pm) < 100:
    print("  WARNING: Very few Mar-May records. Including January as fallback.")
    pm = w[w["month"].isin([1, 3, 4, 5])].copy()
    print(f"  Pre-monsoon (Jan+Mar-May) records: {len(pm)}")

# ── 3. Aggregate: one row per well-year ──
# GWL = depth below ground (metres). Larger = deeper = worse.
agg = pm.groupby([WID_COL, "year"]).agg(
    gwl   = (LEVEL_COL, "mean"),
    lat   = (LAT_COL,   "first"),
    lon   = (LON_COL,   "first"),
).reset_index()
agg = agg.rename(columns={WID_COL: "well_id"})
print(f"  Aggregated well-year rows: {len(agg)}")
print(f"  Unique wells: {agg['well_id'].nunique()}")
print(f"  Years: {sorted(agg['year'].unique())}")
print(f"  GWL stats:\n{agg['gwl'].describe()}")

# ── 4. Spatial join: keep only wells inside the 54 study taluks ──
print("\nSpatial joining wells to labeled taluks...")
gdf = gpd.GeoDataFrame(
    agg,
    geometry=gpd.points_from_xy(agg["lon"], agg["lat"]),
    crs="EPSG:4326"
).to_crs("EPSG:32643")

taluk = gpd.read_file("labeled_taluks.gpkg")
if taluk.crs.to_epsg() != 32643:
    taluk = taluk.to_crs("EPSG:32643")

gdf = gpd.sjoin(
    gdf,
    taluk[["key", "label_binary", "geometry"]],
    how="inner",
    predicate="within"
).rename(columns={"key": "taluk_id"})

# Add UTM coordinates for later pixel lookup
gdf["x"] = gdf.geometry.x
gdf["y"] = gdf.geometry.y

print(f"  Wells inside study area: {gdf['well_id'].nunique()}")
print(f"  Well-year samples inside study area: {len(gdf)}")

# ── 5. Define "stressed" label ──
# Two options are provided. Use the one that makes more sense for your data.

# OPTION A: Global quantile (simple baseline)
# "stressed" = GWL deeper than the 66th percentile across all wells
threshold_global = gdf["gwl"].quantile(0.66)
gdf["stressed_global"] = (gdf["gwl"] > threshold_global).astype(int)

# OPTION B: Within-taluk z-score (recommended — avoids geology bias)
# A well is "stressed" if its GWL is >1 std deeper than its taluk's mean
taluk_stats = gdf.groupby("taluk_id")["gwl"].agg(["mean", "std"]).rename(
    columns={"mean": "taluk_mean", "std": "taluk_std"}
)
gdf = gdf.merge(taluk_stats, left_on="taluk_id", right_index=True)
gdf["gwl_zscore"] = (gdf["gwl"] - gdf["taluk_mean"]) / gdf["taluk_std"].clip(lower=0.01)
gdf["stressed_local"] = (gdf["gwl_zscore"] > 1.0).astype(int)

# DEFAULT: use the local (within-taluk) version
gdf["stressed"] = gdf["stressed_local"]

print(f"\n  Stressed threshold (global 66th pctl): {threshold_global:.2f} m")
print(f"  Stressed counts (global):  {gdf['stressed_global'].value_counts().to_dict()}")
print(f"  Stressed counts (local):   {gdf['stressed_local'].value_counts().to_dict()}")

# ── 6. Save ──
out_cols = ["well_id", "x", "y", "year", "gwl", "stressed", "taluk_id", "label_binary"]
gdf[out_cols].to_parquet("wells_premonsoon.parquet", index=False)
print(f"\nSaved wells_premonsoon.parquet ({len(gdf)} rows)")

# ── 7. Summary per taluk ──
print("\n── Wells per taluk ──")
summary = gdf.groupby(["taluk_id", "label_binary"]).agg(
    n_wells=("well_id", "nunique"),
    mean_gwl=("gwl", "mean"),
    n_stressed=("stressed", "sum"),
).reset_index()
print(summary.sort_values("n_wells", ascending=False).to_string(index=False))

print("\n── Samples per year ──")
print(gdf.groupby("year").size().to_string())
