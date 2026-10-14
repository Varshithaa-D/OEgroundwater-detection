"""
stage1_classical_ml.py - Classical Machine Learning Benchmarks (PyTorch-Free)
==============================================================================
Evaluates classical ML models on 10-year multi-spectral satellite features (26 bands)
using 25.0 km Spatially Buffered Leave-One-Taluk-Out (LOTO) cross-validation.

Models:
  1. kNN_coords (Spatial Negative Control)
  2. LogisticRegression (L2 regularized)
  3. Linear_SVM (Support Vector Classifier)
  4. RBF_SVM (Radial Basis Function Kernel)
  5. RandomForest (Hydrological Ensemble, max_depth=3)
  6. ExtraTrees (Extremely Randomized Trees)
  7. HistGradBoost (Histogram Gradient Boosted Decision Trees)
"""

import os, json, random, argparse, warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ----------------------------------------------------------------------------- CONFIG
NPZ_PATH = "taluk_temporal_patches.npz"
INDEX_PATH = "index.parquet"
GPKG_PATH = "labeled_taluks.gpkg"
GPKG_TABLE, GPKG_KEY = "labeled_taluks", "key"
TALUK_COL, LABEL_COL = "taluk_id", "taluk_label"
N_YEARS, N_BANDS = 10, 13
SEED = 42
BUFFER_KM = 25.0
REF_LAT = 15.0
N_BOOT = 2000
N_PERM = 50
OUT_DIR = os.path.join("runs", "benchmarks")

def load_data(extra_neg=(-1,)):
    if os.path.exists(NPZ_PATH):
        d = np.load(NPZ_PATH, allow_pickle=True)
        mask = np.isin(d["labels"], [0, 1] + list(extra_neg))
        X = d["X"][mask].astype(np.float32)
        X[~np.isfinite(X)] = np.nan
        raw_labels = d["labels"][mask].astype(int)
        taluk_ids = d["taluk_ids"][mask].astype(str)
    else:
        seq_cols = [f"seq_y{y}_b{b}" for y in range(N_YEARS) for b in range(N_BANDS)]
        df = pd.read_parquet(INDEX_PATH, columns=[TALUK_COL, LABEL_COL] + seq_cols)
        df = df[df[LABEL_COL].isin([0, 1] + list(extra_neg))].reset_index(drop=True)
        raw_labels = df[LABEL_COL].to_numpy().astype(int)
        taluk_ids = df[TALUK_COL].astype(str).to_numpy()
        X = df[seq_cols].to_numpy(dtype=np.float32, copy=True).reshape(-1, N_YEARS, N_BANDS)
        X[~np.isfinite(X)] = np.nan

    taluks = np.array(sorted(np.unique(taluk_ids)))
    lab = np.array([int(raw_labels[taluk_ids == t][0] == 1) for t in taluks])
    cat = np.array([int(raw_labels[taluk_ids == t][0]) for t in taluks])
    codes = pd.Categorical(taluk_ids, categories=taluks).codes
    idx_by_taluk = {t: np.where(codes == i)[0] for i, t in enumerate(taluks)}
    return X, taluks, lab, cat, idx_by_taluk

def load_geometry(gpkg_path, taluks):
    import sqlite3, struct
    con = sqlite3.connect(gpkg_path)
    cur = con.cursor()
    cur.execute("SELECT column_name, srs_id FROM gpkg_geometry_columns WHERE table_name=?", (GPKG_TABLE,))
    geom_col, srs = cur.fetchone()
    cur.execute(f'SELECT "{GPKG_KEY}", "{geom_col}" FROM "{GPKG_TABLE}"')
    boxes = {}
    for k, blob in cur.fetchall():
        if blob is None or blob[:2] != b"GP": continue
        flags = blob[3]
        if ((flags >> 1) & 7) == 0: continue
        minx, maxx, miny, maxy = struct.unpack("<dddd" if flags & 1 else ">dddd", blob[8:40])
        boxes[str(k)] = (minx, maxx, miny, maxy)
    con.close()

    B = np.array([boxes[t] for t in taluks], dtype=float)
    if srs in (4326, 4269, 4258):
        kx, ky = 111.32 * np.cos(np.radians(REF_LAT)), 110.57
        B = B * np.array([kx, kx, ky, ky])
    else:
        B = B / 1000.0
    cent = np.stack([(B[:, 0] + B[:, 1]) / 2, (B[:, 2] + B[:, 3]) / 2], axis=1)
    dx = np.maximum(0, np.maximum(B[:, None, 0] - B[None, :, 1], B[None, :, 0] - B[:, None, 1]))
    dy = np.maximum(0, np.maximum(B[:, None, 2] - B[None, :, 3], B[None, :, 2] - B[:, None, 3]))
    return np.sqrt(dx ** 2 + dy ** 2), cent

def taluk_feature_table(X, idx_by_taluk, taluks):
    yrs = np.arange(N_YEARS, dtype=float)
    rows = []
    for t in taluks:
        yearly = np.nanmean(X[idx_by_taluk[t]], axis=0)
        mean_ = np.nanmean(yearly, axis=0)
        slope = np.full(N_BANDS, np.nan)
        for b in range(N_BANDS):
            m = np.isfinite(yearly[:, b])
            if m.sum() >= 3:
                slope[b] = np.polyfit(yrs[m], yearly[m, b], 1)[0]
        rows.append(np.concatenate([mean_, slope]))
    return np.array(rows, dtype=np.float64)

def build_models():
    return {
        "LogisticRegression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=SEED)
        ),
        "Linear_SVM": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            SVC(kernel="linear", C=0.1, class_weight="balanced", probability=True, random_state=SEED)
        ),
        "RBF_SVM": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True, random_state=SEED)
        ),
        "RandomForest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=100, max_depth=3, class_weight="balanced", random_state=SEED)
        ),
        "ExtraTrees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(n_estimators=100, max_depth=3, class_weight="balanced", random_state=SEED)
        ),
        "HistGradBoost": HistGradientBoostingClassifier(
            max_depth=3, class_weight="balanced", random_state=SEED
        )
    }

def run_loto(taluks, lab, D, cent, F_tab, buffer_km=BUFFER_KM):
    n = len(taluks)
    model_defs = build_models()
    model_names = ["kNN_coords"] + list(model_defs.keys())
    scores = {m: np.full(n, np.nan) for m in model_names}
    n_purged = (D <= buffer_km).sum(1) - 1

    for i in range(n):
        tr_idx = np.where(D[i] > buffer_km)[0]
        y_tr = lab[tr_idx]
        if min((y_tr == 0).sum(), (y_tr == 1).sum()) < 2:
            continue

        # 1. Negative Spatial Control (GPS Centroid k-NN)
        knn = KNeighborsClassifier(n_neighbors=min(5, len(tr_idx)), weights="distance")
        knn.fit(cent[tr_idx], y_tr)
        scores["kNN_coords"][i] = float(knn.predict_proba(cent[[i]])[0, 1])

        # 2. Classical Models on Feature Table
        for name, clf in model_defs.items():
            clf.fit(F_tab[tr_idx], y_tr)
            scores[name][i] = float(clf.predict_proba(F_tab[[i]])[0, 1])

    return scores, n_purged

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
    print("=" * 72)
    print("  STAGE 1: CLASSICAL MACHINE LEARNING BENCHMARK (PyTorch-Free)")
    print("=" * 72)
    t0 = time.time()

    X, taluks, lab, cat, idx_by_taluk = load_data(extra_neg=(-1,))
    D, cent = load_geometry(GPKG_PATH, taluks)
    F_tab = taluk_feature_table(X, idx_by_taluk, taluks)

    n_safe = int((lab == 0).sum())
    n_oe = int((lab == 1).sum())
    print(f"Loaded: {len(taluks)} Taluks | Safe/Negatives={n_safe} | Over-Exploited={n_oe}")
    print(f"Features: 26 (13 decadal means + 13 decadal trend slopes)")
    print(f"Protocol: 25.0 km Spatially Buffered Leave-One-Taluk-Out")

    print("\n[1/3] Running 54 LOTO Folds across 7 Classical Models ...")
    scores, n_purged = run_loto(taluks, lab, D, cent, F_tab, buffer_km=BUFFER_KM)
    print(f"Completed in {time.time() - t0:.2f} seconds.")

    print("\n[2/3] Evaluating Models & Computing Bootstrap 95% CIs ...")
    boots = stratified_bootstrap(lab, scores)
    
    print("\n" + "=" * 72)
    print(f"{'Model':<22} {'ROC-AUC':<9} {'95% CI':<18} {'Accuracy':<10} {'Balanced Acc'}")
    print("-" * 72)

    summary_results = {}
    for name, s_arr in scores.items():
        auc = roc_auc_score(lab, s_arr)
        lo, hi = ci(boots[name])
        pred = (s_arr >= 0.50).astype(int)
        acc = (pred == lab).mean() * 100
        bacc = balanced_accuracy_score(lab, pred) * 100
        tn, fp, fn, tp = confusion_matrix(lab, pred, labels=[0, 1]).ravel()
        print(f"{name:<22} {auc*100:5.1f}%    [{lo*100:4.1f}%, {hi*100:4.1f}%]   {acc:5.1f}%     {bacc:5.1f}%")
        summary_results[name] = {
            "auc": float(auc), "ci_lo": float(lo), "ci_hi": float(hi),
            "acc": float(acc), "bacc": float(bacc), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        }
    print("=" * 72)

    # Save scores
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "stage1_classical_scores.csv")
    df_out = pd.DataFrame({"taluk": taluks, "label": lab, "n_purged": n_purged, **scores})
    df_out.to_csv(out_csv, index=False)
    
    with open(os.path.join(OUT_DIR, "stage1_summary.json"), "w") as f:
        json.dump(summary_results, f, indent=2)

    print(f"\n[3/3] Saved results to -> {out_csv}")
    print(f"Total Stage 1 execution time: {time.time() - t0:.2f} seconds.")

if __name__ == "__main__":
    main()
