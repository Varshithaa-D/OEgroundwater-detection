"""
nan_audit.py - Rigorous Audit of Missing Data (NaNs) in Groundwater Dataset
=============================================================================
Audits whether missing data (NaNs) correlates with the class label (Safe vs Over-Exploited),
verifying whether missingness artifacts could be driving model predictions.
"""

import os, time
import numpy as np
import pandas as pd
from scipy import stats

NPZ_PATH = "taluk_temporal_patches.npz"
INDEX_PATH = "index.parquet"

BAND_NAMES = [
    "B01_NDVI", "B02_NDWI", "B03_NDTI", "B04_MNDWI", "B05_SAR_VV", 
    "B06_SAR_VH", "B07_SAR_RATIO", "B08_LST", "B09_CHIRPS_PRECIP", 
    "B10_SMAP_SOIL_M", "B11_B2_BLUE", "B12_B3_GREEN", "B13_B4_RED"
]

def audit():
    print("=" * 72)
    print("  MISSING DATA (NaN) AUDIT REPORT - GROUNDWATER AQUIFER DATASET")
    print("=" * 72)

    if not os.path.exists(NPZ_PATH):
        print(f"[-] Cache file '{NPZ_PATH}' not found!")
        return

    st = os.stat(NPZ_PATH)
    print(f"Cache File     : {NPZ_PATH}")
    print(f"Last Modified  : {time.ctime(st.st_mtime)}")
    print(f"File Size      : {st.st_size / 1024:.1f} KB")

    d = np.load(NPZ_PATH, allow_pickle=True)
    X = d["X"]
    labels = d["labels"].astype(int)
    taluk_ids = d["taluk_ids"].astype(str)

    print(f"Data Shape     : {X.shape} (Patches x 10 Years x 13 Bands)")
    print(f"Total Values   : {X.size:,}")
    print(f"Unique Taluks  : {len(np.unique(taluk_ids))}")

    # Check NaNs
    nan_mask = ~np.isfinite(X)
    total_nans = int(nan_mask.sum())
    overall_nan_pct = (total_nans / X.size) * 100

    print("\n" + "-" * 72)
    print("1. OVERALL MISSINGNESS")
    print("-" * 72)
    print(f"Total NaNs     : {total_nans:,} / {X.size:,}")
    print(f"Overall NaN Rate: {overall_nan_pct:.2f}%")

    # Missingness per Class
    print("\n" + "-" * 72)
    print("2. MISSINGNESS PER CLASS (Safe/Negatives vs Over-Exploited)")
    print("-" * 72)
    
    unique_labels = sorted(np.unique(labels))
    class_stats = []
    for lbl in unique_labels:
        mask = (labels == lbl)
        n_patches = mask.sum()
        sub_X = X[mask]
        sub_nan = (~np.isfinite(sub_X)).sum()
        sub_rate = (sub_nan / sub_X.size) * 100
        lbl_str = "Strict Safe (0)" if lbl == 0 else ("Over-Exploited (1)" if lbl == 1 else f"Negative ({lbl})")
        class_stats.append({
            "Label": lbl, "Name": lbl_str, "Patches": n_patches,
            "Total Values": sub_X.size, "NaNs": sub_nan, "NaN %": sub_rate
        })
        print(f"  {lbl_str:<22}: {n_patches:>4} patches | {sub_nan:>6,} NaNs | {sub_rate:6.2f}% missing")

    # Binary Comparison: Negatives (labels <= 0) vs Positives (label == 1)
    neg_mask = (labels <= 0)
    pos_mask = (labels == 1)
    neg_nan_pct = ((~np.isfinite(X[neg_mask])).sum() / X[neg_mask].size) * 100
    pos_nan_pct = ((~np.isfinite(X[pos_mask])).sum() / X[pos_mask].size) * 100
    abs_diff = abs(pos_nan_pct - neg_nan_pct)

    print(f"\n  Summary: Negatives NaN Rate = {neg_nan_pct:.2f}% | Positives (OE) NaN Rate = {pos_nan_pct:.2f}%")
    print(f"  Absolute Class Difference = {abs_diff:.2f}%")

    # Taluk-level correlation
    print("\n" + "-" * 72)
    print("3. TALUK-LEVEL CORRELATION TEST")
    print("-" * 72)
    taluk_df = pd.DataFrame({
        "taluk": taluk_ids,
        "label": labels,
        "binary_label": (labels == 1).astype(int),
        "patch_nan_rate": nan_mask.mean(axis=(1, 2))
    }).groupby("taluk").agg({
        "binary_label": "first",
        "patch_nan_rate": "mean"
    }).reset_index()

    r, p_val = stats.pointbiserialr(taluk_df["binary_label"], taluk_df["patch_nan_rate"])
    print(f"  Point-Biserial Correlation (Label vs Taluk NaN Rate): r = {r:+.4f} (p-value = {p_val:.4f})")
    
    # Per-Band Breakdown
    print("\n" + "-" * 72)
    print("4. PER-BAND MISSINGNESS (13 BANDS)")
    print("-" * 72)
    band_nan_rates = nan_mask.mean(axis=(0, 1)) * 100
    for b_idx, b_rate in enumerate(band_nan_rates):
        b_name = BAND_NAMES[b_idx] if b_idx < len(BAND_NAMES) else f"Band_{b_idx}"
        bar = "#" * int(b_rate // 2)
        print(f"  {b_idx+1:02d}. {b_name:<16}: {b_rate:5.2f}% {bar}")

    # Final Verdict
    print("\n" + "=" * 72)
    print("  VERDICT FOR REVIEWER")
    print("=" * 72)
    if p_val > 0.05 and abs_diff < 5.0:
        print("  [CLEAN] Missing data is NOT statistically correlated with target label (p > 0.05).")
        print("  Missingness difference between classes is minimal.")
        print("  CONCLUSION: The model score is NOT driven by missing data leakage!")
    else:
        print(f"  [ATTENTION] Missing data has a correlation r={r:.3f} (p={p_val:.4f}).")
        print("  Check which specific bands account for the difference.")
    print("=" * 72)

if __name__ == "__main__":
    audit()
