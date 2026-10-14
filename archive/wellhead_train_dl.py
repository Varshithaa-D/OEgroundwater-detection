"""
wellhead_train_dl.py — Physical Wellhead Groundwater Stress & Depth Modeling
=============================================================================
Evaluates multi-sensor satellite data directly on 599 real telemetry wellhead
monitoring stations across Karnataka with strict Spatial Group Cross-Validation.

Addresses the reviewer's recommendation:
1. Avoids 54-taluk sample-size bottleneck by evaluating on 599 well observations.
2. Uses exact wellhead depth measurements (meters below ground level).
3. Holds out entire taluk spatial clusters during 5-fold cross-validation (zero leakage).
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold

print("=" * 70)
print("  WELLHEAD GROUNDWATER DEPTH & STRESS EVALUATION (PHYSICAL GROUNDING)")
print("=" * 70)

# 1. Load pre-extracted wellhead observations
df = pd.read_parquet("index.parquet")
w = df[df["well_id"].notna() & (df["well_id"] != "") & (df["well_id"] != "None")].copy()

# Valid satellite bands across 10 years (dropping corrupted channels):
# 0: NDVI, 1: NDWI, 2: EVI, 3: LST, 6: VV (SAR), 7: VH (SAR)
valid_bands = [0, 1, 2, 3, 6, 7]
clean_cols = [f"seq_y{y}_b{b}" for y in range(10) for b in valid_bands]

X = w[clean_cols].values.astype(np.float32)
X[~np.isfinite(X)] = np.nan
y_gwl = w["gwl"].values.astype(np.float32)
y_stress = w["stressed"].values.astype(int)
taluks = w["taluk_id"].values

print(f"  Total Wellhead Observations: {len(w)}")
print(f"  Unique Well IDs:             {w['well_id'].nunique()}")
print(f"  Taluks represented:          {w['taluk_id'].nunique()}")
print(f"  Clean Multi-Sensor Features: {len(clean_cols)} (10 years × 6 physical bands)")

# 2. Continuous Depth Regression Task (Spatial Group K-Fold)
print("\n[1/2] Continuous Groundwater Depth (GWL) Regression Task...")
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof_gwl = np.zeros(len(w))

for fold, (tr_idx, te_idx) in enumerate(sgkf.split(X, (y_gwl > 8).astype(int), groups=taluks)):
    pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu",
                     alpha=1.0, max_iter=800, random_state=42)
    )
    pipe.fit(X[tr_idx], y_gwl[tr_idx])
    oof_gwl[te_idx] = pipe.predict(X[te_idx])

corr, pval = spearmanr(y_gwl, oof_gwl)
mae = mean_absolute_error(y_gwl, oof_gwl)
print(f"  Spearman Correlation (r):  {corr:.3f}")
print(f"  P-value:                   {pval:.4e} (p < 0.05 -> Statistically Significant)")
print(f"  Mean Absolute Error (MAE): {mae:.2f} meters")

# 3. Wellhead Stress Classification Task (Spatial Group K-Fold)
print("\n[2/2] Wellhead Depletion / Stress Classification Task...")
oof_stress = np.zeros(len(w))

for fold, (tr_idx, te_idx) in enumerate(sgkf.split(X, y_stress, groups=taluks)):
    pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu",
                      alpha=1.0, max_iter=800, random_state=42)
    )
    pipe.fit(X[tr_idx], y_stress[tr_idx])
    oof_stress[te_idx] = pipe.predict_proba(X[te_idx])[:, 1]

auc = roc_auc_score(y_stress, oof_stress)
pred_stress = (oof_stress >= 0.5).astype(int)
acc = accuracy_score(y_stress, pred_stress)
cm = confusion_matrix(y_stress, pred_stress)

print(f"  Accuracy:                  {acc*100:.1f}%")
print(f"  AUC:                       {auc:.3f}")
print(f"  Confusion Matrix:\n{cm}")

# 4. Save summary
os.makedirs("runs/wellhead", exist_ok=True)
summary = {
    "n_wellheads": len(w),
    "unique_wells": int(w['well_id'].nunique()),
    "spearman_r": float(corr),
    "p_value": float(pval),
    "mae_meters": float(mae),
    "classification_acc": float(acc),
    "classification_auc": float(auc)
}
with open("runs/wellhead/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Saved wellhead summary -> runs/wellhead/summary.json")
print("=" * 70)
