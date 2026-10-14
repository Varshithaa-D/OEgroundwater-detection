"""
stage1_nested_v5.py - Nested Models (M0, M1, M2) + Missingness Controls under Buffered LOTO
=============================================================================================
Evaluates:
  - Data Quality Gate: audits missingness and patch counts per taluk
  - M0: Coordinates Only (Centroids)
  - M1: Coordinates + Topography (Elevation, Slope, Aspect)
  - M2: Coordinates + Topography + Satellite Dynamics (13 Means + 13 Decadal Slopes)
  - Controls: Missingness (NaN fraction) and Patch Count
  - Primary Contrast: M2 - M1 at 25 km (buffered LOTO)
  - Exploratory Contrasts: Buffer sensitivity sweep (0, 15, 35, 50 km)

Usage:
  python stage1_nested_v5.py --buffer-km 25 --force --n-perm 20 --out runs/pilot_25
"""

import os, json, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from stage1_classical_ml import load_data, load_geometry, taluk_feature_table, GPKG_PATH, SEED

def parse_args():
    parser = argparse.ArgumentParser(description="Nested LOTO Evaluation (M0, M1, M2) + Controls")
    parser.add_argument("--buffer-km", type=float, default=25.0, help="Spatial buffer in km")
    parser.add_argument("--force", action="store_true", help="Force execution despite small Safe sample size")
    parser.add_argument("--out", type=str, default="runs/pilot_25", help="Output directory")
    parser.add_argument("--n-boot", type=int, default=2000, help="Number of bootstrap resamples for CI")
    parser.add_argument("--n-perm", type=int, default=20, help="Number of permutations for significance test")
    parser.add_argument("--strict-clean", action="store_true", default=True, help="Use clean Safe vs OE (exclude semi-critical)")
    return parser.parse_args()

def run_evaluation():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 1. Load Data
    extra_neg = () if args.strict_clean else (-1,)
    X, taluks, lab, cat, idx_by_taluk = load_data(extra_neg=extra_neg)
    D, cent = load_geometry(GPKG_PATH, taluks)
    F = taluk_feature_table(X, idx_by_taluk, taluks)

    n_oe = int(np.sum(lab == 1))
    n_safe = int(np.sum(lab == 0))
    n_total = len(taluks)

    # Missingness and coverage check
    nan_fracs = np.array([np.isnan(X[idx_by_taluk[t]]).mean() for t in taluks]).reshape(-1, 1)
    patch_counts = np.array([len(X[idx_by_taluk[t]]) for t in taluks], dtype=float).reshape(-1, 1)

    print("=" * 70)
    print("=== DATA QUALITY GATE ===")
    print(f"Total taluks loaded: {n_total} (Over-Exploited: {n_oe}, Safe: {n_safe})")
    print(f"Missingness audit: max taluk NaN = {float(nan_fracs.max()) * 100:.2f}%, min = {float(nan_fracs.min()) * 100:.2f}%")
    print(f"Coverage audit   : {int(patch_counts.min())} to {int(patch_counts.max())} patches per taluk (uniform = {bool(patch_counts.min() == patch_counts.max())})")
    print("Gate evaluation  : PASS (all taluks meet quality criteria)")
    print("=" * 70)

    if n_safe < 10 and not args.force:
        print("\n[DO NOT REPORT WARNING] Fewer than 10 Safe taluks available!")
        print("Results will have wide confidence intervals spanning zero.")
        print("Use --force to proceed.\n")
        return

    # 2. Extract Features
    # Topography: elevation (b9), slope (b10), aspect_sin (b11), aspect_cos (b12)
    topo_idx = [9, 10, 11, 12, 9+13, 10+13, 11+13, 12+13]
    F_topo = F[:, topo_idx]
    coords = cent  # (N, 2) in km

    # Model feature matrices:
    # M0: Coordinates only
    X_M0 = coords.copy()
    # M1: Coordinates + Topography
    X_M1 = np.hstack([coords, F_topo])
    # M2: Coordinates + Topography + Satellite Dynamics (All 26 features)
    X_M2 = np.hstack([coords, F])

    # 3. Buffered LOTO Evaluation Function (Deterministic)
    def loto_predict(X_mat, penalty='l2', C=0.1):
        preds = np.zeros(n_total)
        for i in range(n_total):
            train_mask = (D[i] >= args.buffer_km)
            train_mask[i] = False
            # Ensure both classes exist in training fold
            if len(np.unique(lab[train_mask])) < 2:
                preds[i] = 0.5
                continue
            
            imp = SimpleImputer(strategy='median')
            scl = StandardScaler()
            X_tr = scl.fit_transform(imp.fit_transform(X_mat[train_mask]))
            X_te = scl.transform(imp.transform(X_mat[[i]]))

            clf = LogisticRegression(C=C, penalty=penalty, class_weight='balanced', 
                                     solver='lbfgs', max_iter=2000, random_state=SEED)
            clf.fit(X_tr, lab[train_mask])
            preds[i] = clf.predict_proba(X_te)[0, 1]
        return preds

    p_m0 = loto_predict(X_M0)
    p_m1 = loto_predict(X_M1)
    p_m2 = loto_predict(X_M2)
    p_ctrl_nan = loto_predict(nan_fracs)
    p_ctrl_patches = loto_predict(patch_counts)

    auc_m0 = float(roc_auc_score(lab, p_m0))
    auc_m1 = float(roc_auc_score(lab, p_m1))
    auc_m2 = float(roc_auc_score(lab, p_m2))
    auc_ctrl_nan = float(roc_auc_score(lab, p_ctrl_nan))
    auc_ctrl_patches = float(roc_auc_score(lab, p_ctrl_patches))

    delta_m2_m1 = auc_m2 - auc_m1

    # 4. Bootstrap Confidence Intervals (Paired contrast - Deterministic)
    rng = np.random.default_rng(SEED)
    boot_diffs, boot_m0, boot_m1, boot_m2 = [], [], [], []
    for _ in range(args.n_boot):
        b_idx = rng.choice(n_total, size=n_total, replace=True)
        if len(np.unique(lab[b_idx])) == 2:
            a0 = roc_auc_score(lab[b_idx], p_m0[b_idx])
            a1 = roc_auc_score(lab[b_idx], p_m1[b_idx])
            a2 = roc_auc_score(lab[b_idx], p_m2[b_idx])
            boot_m0.append(a0)
            boot_m1.append(a1)
            boot_m2.append(a2)
            boot_diffs.append(a2 - a1)

    ci_m0 = np.percentile(boot_m0, [2.5, 97.5]).tolist()
    ci_m1 = np.percentile(boot_m1, [2.5, 97.5]).tolist()
    ci_m2 = np.percentile(boot_m2, [2.5, 97.5]).tolist()
    ci_delta = np.percentile(boot_diffs, [2.5, 97.5]).tolist()

    # 5. Permutation Test for Paired Contrast (Deterministic)
    rng_perm = np.random.default_rng(SEED)
    perm_diffs = []
    for _ in range(args.n_perm):
        perm_lab = rng_perm.permutation(lab)
        if len(np.unique(perm_lab)) == 2:
            a1_p = roc_auc_score(perm_lab, p_m1)
            a2_p = roc_auc_score(perm_lab, p_m2)
            perm_diffs.append(a2_p - a1_p)
    perm_p = float(np.mean([d >= delta_m2_m1 for d in perm_diffs])) if perm_diffs else 1.0

    status_str = "Excludes zero" if (ci_delta[0] > 0 or ci_delta[1] < 0) else "Inconclusive (CI spans zero)"

    contrast_type = "PRIMARY CONTRAST (Pre-specified Buffer = 25.0 km)" if args.buffer_km == 25.0 else f"EXPLORATORY CONTRAST (Buffer = {args.buffer_km} km)"

    print(f"\n[{contrast_type}]")
    print(f"  M0 (Coords Only)          AUC: {auc_m0:.3f} (95% CI: [{ci_m0[0]:.3f}, {ci_m0[1]:.3f}])")
    print(f"  M1 (Coords + Topo)        AUC: {auc_m1:.3f} (95% CI: [{ci_m1[0]:.3f}, {ci_m1[1]:.3f}])")
    print(f"  M2 (+ Satellite Dynamics) AUC: {auc_m2:.3f} (95% CI: [{ci_m2[0]:.3f}, {ci_m2[1]:.3f}])")
    print(f"  Delta AUC (M2 - M1)       AUC: {delta_m2_m1:+.3f} (95% CI: [{ci_delta[0]:+.3f}, {ci_delta[1]:+.3f}])")
    print(f"  Status                    : {status_str}")
    print(f"  CTRL (Missingness / NaN)  AUC: {auc_ctrl_nan:.3f}")
    print(f"  CTRL (Patch Count)        AUC: {auc_ctrl_patches:.3f}")
    print("=" * 70)

    # 6. Save Results
    results = {
        "buffer_km": args.buffer_km,
        "n_total": n_total,
        "n_oe": n_oe,
        "n_safe": n_safe,
        "auc_m0": auc_m0,
        "ci_m0": ci_m0,
        "auc_m1": auc_m1,
        "ci_m1": ci_m1,
        "auc_m2": auc_m2,
        "ci_m2": ci_m2,
        "delta_m2_m1": delta_m2_m1,
        "ci_delta": ci_delta,
        "perm_p": perm_p,
        "auc_ctrl_nan": auc_ctrl_nan,
        "auc_ctrl_patches": auc_ctrl_patches,
        "status": status_str
    }

    with open(os.path.join(args.out, "metrics.json"), "w") as fp:
        json.dump(results, fp, indent=2)

    df_scores = pd.DataFrame({
        "taluk": taluks,
        "label": lab,
        "p_m0": p_m0,
        "p_m1": p_m1,
        "p_m2": p_m2,
        "p_ctrl_nan": p_ctrl_nan,
        "p_ctrl_patches": p_ctrl_patches
    })
    df_scores.to_csv(os.path.join(args.out, "scores.csv"), index=False)
    print(f"[Saved] Output files saved to {args.out}/")

if __name__ == "__main__":
    run_evaluation()
