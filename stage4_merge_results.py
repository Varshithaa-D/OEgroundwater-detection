"""
stage4_merge_results.py - Final Unified Benchmark Comparison & Paper Table
=============================================================================
Merges predictions across:
  - Stage 1: Classical ML (Logistic Regression, SVM, Random Forest, etc.)
  - Stage 2: Bio-Inspired Feature Selection (GA + SVM, PSO + SVM, GA + LR)
  - Stage 3: Deep Learning (Temporal Bidirectional GRU)
  - Spatial Negative Control (k-NN on Coordinates)

Calculates:
  - ROC-AUC with 95% Stratified Bootstrap Confidence Intervals (2,000 resamples)
  - Accuracy and Balanced Accuracy (at 0.50 threshold)
  - Confusion Matrix (TN, FP, FN, TP)
  - Paired Bootstrap Hypothesis Testing (Bi-GRU vs Each Model)
  - Prints the complete Results Table ready for project reports / presentations.
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix

S1_PATH = os.path.join("runs", "benchmarks", "stage1_classical_scores.csv")
S2_PATH = os.path.join("runs", "benchmarks", "stage2_bio_scores.csv")
S3_PATH = os.path.join("runs", "clean_dl_v4", "loto_scores.csv")
OUT_DIR = os.path.join("runs", "benchmarks")

N_BOOT = 2000
SEED = 42

def stratified_bootstrap(y, scores, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    out = {k: np.empty(n_boot) for k in scores}
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for k, s in scores.items():
            out[k][b] = roc_auc_score(y[idx], s[idx])
    return out

def ci(arr):
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

def main():
    print("=" * 84)
    print("  STAGE 4: UNIFIED MULTI-MODEL BENCHMARK & COMPARISON TABLE")
    print("=" * 84)

    # 1. Load Data
    if not os.path.exists(S1_PATH) or not os.path.exists(S2_PATH):
        raise FileNotFoundError("Stage 1 or Stage 2 outputs missing!")

    df1 = pd.read_csv(S1_PATH)
    df2 = pd.read_csv(S2_PATH)
    df3 = pd.read_csv(S3_PATH) if os.path.exists(S3_PATH) else None

    # Merge by taluk
    merged = df1.copy()
    for col in df2.columns:
        if col not in merged.columns and col in ["GA_SVM", "PSO_SVM", "GA_LogisticReg"]:
            merged[col] = df2[col].values

    s3_bench = os.path.join("runs", "benchmarks", "stage3_deep_learning_scores.csv")
    if os.path.exists(s3_bench):
        df_bench = pd.read_csv(s3_bench)
        if "DL_Bi_GRU" in df_bench.columns:
            merged["DL_Bi_GRU"] = merged["taluk"].map(dict(zip(df_bench["taluk"], df_bench["DL_Bi_GRU"])))
        if "DL_1D_CNN" in df_bench.columns:
            merged["DL_1D_CNN"] = merged["taluk"].map(dict(zip(df_bench["taluk"], df_bench["DL_1D_CNN"])))
        if "DL_Transformer" in df_bench.columns:
            merged["DL_Transformer"] = merged["taluk"].map(dict(zip(df_bench["taluk"], df_bench["DL_Transformer"])))
    elif df3 is not None and "DL_GRU" in df3.columns:
        # Fallback to previously saved Bi-GRU scores
        m_s3 = dict(zip(df3["taluk"], df3["DL_GRU"]))
        merged["DL_Bi_GRU"] = merged["taluk"].map(m_s3)

    y = merged["label"].values
    taluks = merged["taluk"].values
    n_safe = int((y == 0).sum())
    n_oe = int((y == 1).sum())

    print(f"Evaluated Region : 54 Taluks in Karnataka")
    print(f"Class Breakdown  : {n_safe} Safe/Negatives vs {n_oe} Over-Exploited (440 Pairs)")
    print(f"Purge Protocol   : 25.0 km Geographic Buffer (Zero Spatial Leakage)")
    print(f"Resampling       : 2,000 Stratified Bootstrap Iterations for 95% CIs")

    # Models to evaluate
    model_cols = [
        ("Spatial Control", "kNN_coords", "Geographic k-NN (Coordinates Only)"),
        ("Classical ML",   "LogisticRegression", "Logistic Regression (L2)"),
        ("Classical ML",   "Linear_SVM", "Linear Support Vector Classifier"),
        ("Classical ML",   "RandomForest", "Random Forest (100 Trees)"),
        ("Classical ML",   "ExtraTrees", "Extra-Trees (Extremely Randomized)"),
        ("Classical ML",   "RBF_SVM", "Non-linear RBF Kernel SVM"),
        ("Classical ML",   "HistGradBoost", "Histogram Gradient Boosting"),
        ("Bio-Inspired",   "GA_SVM", "Genetic Algorithm (GA) + SVM"),
        ("Bio-Inspired",   "GA_LogisticReg", "Genetic Algorithm (GA) + LogReg"),
        ("Bio-Inspired",   "PSO_SVM", "Particle Swarm Optimization (PSO) + SVM"),
    ]
    if "DL_Bi_GRU" in merged.columns:
        model_cols.append(("Deep Learning", "DL_Bi_GRU", "Temporal Bidirectional GRU"))
    if "DL_1D_CNN" in merged.columns:
        model_cols.append(("Deep Learning", "DL_1D_CNN", "Temporal 1D-CNN (Kernel=3)"))
    if "DL_Transformer" in merged.columns:
        model_cols.append(("Deep Learning", "DL_Transformer", "Small Temporal Transformer"))

    score_dict = {col: merged[col].values for _, col, _ in model_cols if col in merged.columns}
    boots = stratified_bootstrap(y, score_dict)

    # Build Comparison Table
    table_rows = []
    print("\n" + "=" * 90)
    print(f"{'Category':<16} {'Model':<22} {'ROC-AUC':<9} {'95% CI':<19} {'Acc':<7} {'Bal Acc':<9} {'(TN, FP, FN, TP)'}")
    print("-" * 90)

    for cat_name, col_name, full_name in model_cols:
        if col_name not in score_dict: continue
        s_arr = score_dict[col_name]
        auc = roc_auc_score(y, s_arr)
        lo, hi = ci(boots[col_name])
        pred = (s_arr >= 0.50).astype(int)
        acc = (pred == y).mean() * 100
        bacc = balanced_accuracy_score(y, pred) * 100
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

        print(f"{cat_name:<16} {col_name:<22} {auc*100:5.1f}%    [{lo*100:4.1f}%, {hi*100:4.1f}%]     {acc:5.1f}%   {bacc:5.1f}%    ({tn:2d}, {fp:2d}, {fn:2d}, {tp:2d})")
        table_rows.append({
            "Category": cat_name,
            "Model Code": col_name,
            "Model Name": full_name,
            "ROC_AUC": round(auc * 100, 2),
            "CI_95_Lower": round(lo * 100, 2),
            "CI_95_Upper": round(hi * 100, 2),
            "Accuracy": round(acc, 2),
            "Balanced_Accuracy": round(bacc, 2),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
        })
    print("=" * 90)

    # Paired Bootstrap Differences vs Bi-GRU
    if "DL_Bi_GRU" in score_dict:
        print("\n" + "-" * 90)
        print("  PAIRED BOOTSTRAP SIGNIFICANCE TESTING (Temporal Bi-GRU vs Other Models)")
        print("-" * 90)
        gru_boots = boots["DL_Bi_GRU"]
        for _, col_name, full_name in model_cols:
            if col_name in ["DL_Bi_GRU", "kNN_coords"]: continue
            diff = gru_boots - boots[col_name]
            d_mean = diff.mean()
            d_lo, d_hi = ci(diff)
            if d_lo > 0:
                verdict = "Bi-GRU statistically superior (p < 0.05)"
            elif d_hi < 0:
                verdict = f"{col_name} statistically superior (p < 0.05)"
            else:
                verdict = "Statistically indistinguishable (overlapping CIs)"
            print(f"  Bi-GRU vs {col_name:<20}: Diff = {d_mean*100:+5.1f}%  CI [{d_lo*100:+4.1f}%, {d_hi*100:+4.1f}%]  -> {verdict}")

    # Save outputs
    res_df = pd.DataFrame(table_rows)
    csv_out = os.path.join(OUT_DIR, "final_comparison_table.csv")
    md_out = os.path.join(OUT_DIR, "final_comparison_table.md")
    merged_scores_out = os.path.join(OUT_DIR, "final_merged_scores.csv")

    res_df.to_csv(csv_out, index=False)
    merged.to_csv(merged_scores_out, index=False)
    
    # Save nice markdown table
    sub_df = res_df[["Category", "Model Name", "ROC_AUC", "Accuracy", "Balanced_Accuracy"]]
    header = "| " + " | ".join(sub_df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(sub_df.columns)) + " |"
    rows = ["| " + " | ".join(str(val) for val in row) + " |" for row in sub_df.values]
    md_table = "\n".join([header, sep] + rows)
    with open(md_out, "w") as f:
        f.write("# Groundwater Over-Exploitation Detection - Unified Benchmark Results\n\n")
        f.write(md_table + "\n")

    print(f"\n[SUCCESS] Files saved:")
    print(f"  --> {csv_out}")
    print(f"  --> {md_out}")
    print(f"  --> {merged_scores_out}")
    print("=" * 84)

if __name__ == "__main__":
    main()
