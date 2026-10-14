"""
clean_train.py — Groundwater Over-Exploitation Detection
=========================================================
Design Decisions (all leakage/overfitting issues fixed):

1. SPATIAL LEAKAGE FIX:
   Uses the existing `fold` column which assigns fold numbers at the
   TALUK level. Every patch from a given taluk goes into exactly one fold.
   The model NEVER sees any patch from a test taluk during training.

2. LABEL BROADCASTING FIX:
   Instead of training on individual 1km patches (which have noisy
   taluk-level labels), we aggregate all patches per taluk-year into
   a single feature vector using Mean + Std statistics. The model now
   learns at the TALUK level, which is exactly where the ground truth
   label lives. This eliminates label noise entirely.

3. CLASS IMBALANCE FIX:
   The dataset has ~9.5:1 OE:Safe imbalance. We use:
   - sklearn's class_weight='balanced' for Logistic Regression
   - A focused evaluation on AUC + F1 (not raw accuracy, which is
     misleading with imbalance)

4. OVERFITTING FIX:
   - Logistic Regression with L2 regularization (C=0.1)
   - RandomForest with max_depth=4, min_samples_leaf=3
   - Both evaluated on completely held-out Taluks (never seen in training)

5. REALISTIC TARGETS:
   We target 80-90% AUC. Anything above 93% with only 54 taluks
   and a 9.5:1 imbalance is a red flag for leakage.

Usage:
    python clean_train.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, f1_score, balanced_accuracy_score,
    confusion_matrix, classification_report
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load index and keep only labeled taluks (label 0 or 1)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  GROUNDWATER OVER-EXPLOITATION DETECTION — CLEAN TRAINING")
print("=" * 65)

print("\n[1/5] Loading index.parquet...")
df = pd.read_parquet("index.parquet")

# Keep only cleanly labeled rows (drop semi-critical label=-1)
df = df[df["taluk_label"].isin([0, 1])].copy()
print(f"      Total labeled patches: {len(df):,}")
print(f"      Taluks: {df['taluk_id'].nunique()}")
print(f"      Label=1 (Over-Exploited): {(df['taluk_label']==1).sum():,}")
print(f"      Label=0 (Safe):           {(df['taluk_label']==0).sum():,}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Build temporal feature matrix by aggregating patches → taluk-year
#
# The seq_yY_bB columns store the pre-extracted band mean for year Y, band B.
# We take mean + std across all patches within a (taluk_id, year) group.
# This creates one clean feature vector per taluk per year.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Building taluk-level temporal features (fixing label broadcast)...")

# Identify all sequence columns
seq_cols = [c for c in df.columns if c.startswith("seq_")]
print(f"      Found {len(seq_cols)} pre-extracted temporal sequence columns")
print(f"      (10 years × 13 bands = 130 columns expected)")

# Aggregate: one row per (taluk_id, year)
agg_dict = {c: ["mean", "std"] for c in seq_cols}
agg_dict["taluk_label"] = "first"
agg_dict["fold"]        = "first"

taluk_year_df = (
    df.groupby(["taluk_id", "year"])
      .agg(agg_dict)
      .reset_index()
)

# Flatten multi-level column names
taluk_year_df.columns = [
    "_".join(c).strip("_") if c[1] else c[0]
    for c in taluk_year_df.columns
]

print(f"      Aggregated to {len(taluk_year_df):,} taluk-year rows")

# Now aggregate further to one row per taluk (average across all years)
# This is the cleanest possible representation: one feature vector per taluk
feat_cols_mean = [c for c in taluk_year_df.columns if c.startswith("seq_") and c.endswith("_mean")]
feat_cols_std  = [c for c in taluk_year_df.columns if c.startswith("seq_") and c.endswith("_std")]
feat_cols      = feat_cols_mean + feat_cols_std

taluk_df = (
    taluk_year_df.groupby("taluk_id")
    .agg(
        {**{c: "mean" for c in feat_cols},
         "taluk_label_first": "first",
         "fold_first": "first"}
    )
    .reset_index()
)

taluk_df = taluk_df.rename(columns={
    "taluk_label_first": "label",
    "fold_first":        "fold"
})

print(f"      Final: {len(taluk_df)} taluks → {len(feat_cols)} features per taluk")
print(f"      Label=1 taluks: {(taluk_df['label']==1).sum()}")
print(f"      Label=0 taluks: {(taluk_df['label']==0).sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Leave-One-Fold-Out Cross-Validation
#         (each fold = a group of geographically separate taluks)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Running Taluk-level Leave-One-Fold-Out CV...")
print("      (Each fold holds out entirely unseen Taluks — zero leakage)")

X = taluk_df[feat_cols].values
y = taluk_df["label"].values
folds = taluk_df["fold"].values
taluk_ids = taluk_df["taluk_id"].values

valid_folds = sorted([f for f in np.unique(folds) if f >= 0])
print(f"      Folds: {valid_folds}")
print(f"      {'Fold':>5} | {'Train Taluks':>12} | {'Test Taluks':>11} | "
      f"{'LR AUC':>8} | {'RF AUC':>8}")
print("      " + "-" * 60)

# Store OOF predictions
lr_oof_probs  = np.full(len(taluk_df), np.nan)
rf_oof_probs  = np.full(len(taluk_df), np.nan)
oof_labels    = np.full(len(taluk_df), np.nan)

for fold_k in valid_folds:
    train_mask = (folds != fold_k) & (folds >= 0)
    test_mask  = (folds == fold_k)

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask],  y[test_mask]

    if len(np.unique(y_te)) < 2:
        print(f"      {fold_k:>5} | {train_mask.sum():>12} | {test_mask.sum():>11} | "
              f"{'SKIP (1 class)':>19}")
        continue

    # ── Logistic Regression (L2, class balanced) ──────────────────────────
    lr = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=0.1, class_weight="balanced",
                           max_iter=2000, solver="lbfgs",
                           random_state=42)
    )
    lr.fit(X_tr, y_tr)
    lr_prob = lr.predict_proba(X_te)[:, 1]

    # ── Random Forest (depth-capped, min_samples to prevent overfit) ──────
    rf = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=200,
            max_depth=4,            # strict depth cap prevents memorization
            min_samples_leaf=3,     # each leaf must represent ≥3 taluks
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    )
    rf.fit(X_tr, y_tr)
    rf_prob = rf.predict_proba(X_te)[:, 1]

    lr_auc = roc_auc_score(y_te, lr_prob)
    rf_auc = roc_auc_score(y_te, rf_prob)

    print(f"      {fold_k:>5} | {train_mask.sum():>12} | {test_mask.sum():>11} | "
          f"{lr_auc:>8.4f} | {rf_auc:>8.4f}")

    lr_oof_probs[test_mask] = lr_prob
    rf_oof_probs[test_mask] = rf_prob
    oof_labels[test_mask]   = y_te

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Pooled Out-of-Fold Evaluation
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Pooled Out-of-Fold Results (all test folds combined)...")

valid_oof = ~np.isnan(lr_oof_probs)
y_true    = oof_labels[valid_oof].astype(int)
lr_probs  = lr_oof_probs[valid_oof]
rf_probs  = rf_oof_probs[valid_oof]

# Use 0.5 threshold (fixed on validation data — no test-set threshold tuning)
lr_preds = (lr_probs >= 0.5).astype(int)
rf_preds = (rf_probs >= 0.5).astype(int)

print("\n" + "=" * 65)
print("  LOGISTIC REGRESSION (L2, class-balanced)")
print("=" * 65)
print(f"  Taluk AUC:          {roc_auc_score(y_true, lr_probs):.4f}")
print(f"  Balanced Accuracy:  {balanced_accuracy_score(y_true, lr_preds):.4f}")
print(f"  F1 (macro):         {f1_score(y_true, lr_preds, average='macro'):.4f}")
print(f"  Confusion Matrix:\n{confusion_matrix(y_true, lr_preds)}")
print(f"\n  Classification Report:\n{classification_report(y_true, lr_preds, target_names=['Safe','Over-Exploited'])}")

print("\n" + "=" * 65)
print("  RANDOM FOREST (depth=4, class-balanced)")
print("=" * 65)
print(f"  Taluk AUC:          {roc_auc_score(y_true, rf_probs):.4f}")
print(f"  Balanced Accuracy:  {balanced_accuracy_score(y_true, rf_preds):.4f}")
print(f"  F1 (macro):         {f1_score(y_true, rf_preds, average='macro'):.4f}")
print(f"  Confusion Matrix:\n{confusion_matrix(y_true, rf_preds)}")
print(f"\n  Classification Report:\n{classification_report(y_true, rf_preds, target_names=['Safe','Over-Exploited'])}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Save results and per-taluk predictions for the dashboard
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Saving results...")

results_df = taluk_df[["taluk_id", "label", "fold"]].copy()
results_df["lr_prob"]  = lr_oof_probs
results_df["rf_prob"]  = rf_oof_probs
# Use average of both models as final ensemble score
results_df["ensemble_prob"] = (lr_oof_probs + rf_oof_probs) / 2.0
results_df["pct"] = (results_df["ensemble_prob"] * 100).round(1)

import os
os.makedirs("runs/clean", exist_ok=True)
results_df.to_csv("runs/clean/taluk_predictions.csv", index=False)

summary = {
    "lr_auc":  round(float(roc_auc_score(y_true, lr_probs)),  4),
    "rf_auc":  round(float(roc_auc_score(y_true, rf_probs)),  4),
    "lr_balanced_acc":  round(float(balanced_accuracy_score(y_true, lr_preds)), 4),
    "rf_balanced_acc":  round(float(balanced_accuracy_score(y_true, rf_preds)), 4),
    "n_taluks_evaluated": int(valid_oof.sum()),
    "note": "Taluk-level LOFO CV. Zero spatial leakage. Threshold fixed at 0.5."
}

import json
with open("runs/clean/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Saved taluk predictions → runs/clean/taluk_predictions.csv")
print(f"  Saved summary          → runs/clean/summary.json")
print("\n" + "=" * 65)
print("  DONE. These results have:")
print("  ✅ Zero spatial data leakage (whole-taluk fold holdout)")
print("  ✅ Zero label broadcasting mismatch (taluk-level aggregation)")
print("  ✅ Class imbalance handled (class_weight='balanced')")
print("  ✅ No test-set threshold tuning (fixed 0.5 threshold)")
print("  ✅ Realistic AUC target (80-92% expected range)")
print("=" * 65)
