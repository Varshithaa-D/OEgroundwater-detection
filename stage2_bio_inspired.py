"""
stage2_bio_inspired.py - Bio-Inspired Feature Selection (GA & PSO) Benchmark
=============================================================================
Applies Genetic Algorithm (GA) and Particle Swarm Optimization (PSO) feature selection
nested inside each of the 54 Leave-One-Taluk-Out (LOTO) cross-validation folds.

Protocol:
  - 25.0 km Spatially Buffered LOTO (Zero Spatial Leakage).
  - Feature selection is strictly nested on the fold's training set (using internal 3-fold CV).
  - Selected features are evaluated on the held-out test taluk.
  - Models: GA + SVM, PSO + SVM, GA + LogisticRegression.
"""

import os, json, random, warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
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
    dx = np.maximum(0, np.maximum(B[:, None, 0] - B[None, :, 1], B[None, :, 0] - B[:, None, 1]))
    dy = np.maximum(0, np.maximum(B[:, None, 2] - B[None, :, 3], B[None, :, 2] - B[:, None, 3]))
    return np.sqrt(dx ** 2 + dy ** 2)

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

# ----------------------------------------------------------------------------- BIO-INSPIRED OPTIMIZERS
def evaluate_subset_cv(feat_mask, X_train, y_train, clf_type="svm"):
    if feat_mask.sum() == 0:
        return 0.0
    cols = np.where(feat_mask)[0]
    X_sub = X_train[:, cols]
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    aucs = []
    for tr, val in cv.split(X_sub, y_train):
        if len(np.unique(y_train[val])) < 2: continue
        if clf_type == "svm":
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  SVC(kernel="linear", C=0.1, class_weight="balanced", probability=True, random_state=SEED))
        else:
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  LogisticRegression(C=0.1, class_weight="balanced", max_iter=500, random_state=SEED))
        model.fit(X_sub[tr], y_train[tr])
        probs = model.predict_proba(X_sub[val])[:, 1]
        aucs.append(roc_auc_score(y_train[val], probs))
    return np.mean(aucs) if len(aucs) > 0 else 0.5

def genetic_algorithm_select(X_train, y_train, n_features=26, pop_size=20, n_gen=12, clf_type="svm"):
    rng = np.random.default_rng(SEED)
    pop = rng.integers(0, 2, size=(pop_size, n_features)).astype(bool)
    # Ensure at least 3 features active per individual
    for i in range(pop_size):
        if pop[i].sum() < 3:
            pop[i, rng.choice(n_features, 3, replace=False)] = True

    best_mask = pop[0]
    best_fitness = -1.0

    for gen in range(n_gen):
        fitness = np.array([evaluate_subset_cv(ind, X_train, y_train, clf_type) for ind in pop])
        if fitness.max() > best_fitness:
            best_fitness = fitness.max()
            best_mask = pop[np.argmax(fitness)].copy()

        # Selection (tournament)
        new_pop = []
        for _ in range(pop_size):
            t1, t2 = rng.choice(pop_size, 2, replace=False)
            winner = pop[t1] if fitness[t1] >= fitness[t2] else pop[t2]
            new_pop.append(winner.copy())
        
        # Crossover & Mutation
        for i in range(0, pop_size, 2):
            if rng.random() < 0.8:
                pt = rng.integers(1, n_features)
                new_pop[i][pt:], new_pop[i+1][pt:] = new_pop[i+1][pt:].copy(), new_pop[i][pt:].copy()
            # Mutation (flip bit with prob 0.05)
            mut_mask = rng.random(n_features) < 0.05
            new_pop[i][mut_mask] = ~new_pop[i][mut_mask]
            if new_pop[i].sum() == 0: new_pop[i][rng.choice(n_features, 3, replace=False)] = True
            
        pop = np.array(new_pop)

    return best_mask

def particle_swarm_select(X_train, y_train, n_features=26, n_particles=15, n_iter=10, clf_type="svm"):
    rng = np.random.default_rng(SEED)
    pos = rng.uniform(0.1, 0.9, size=(n_particles, n_features))
    vel = rng.uniform(-0.1, 0.1, size=(n_particles, n_features))
    
    pbest = pos.copy()
    pbest_val = np.array([evaluate_subset_cv(pos[i] > 0.5, X_train, y_train, clf_type) for i in range(n_particles)])
    gbest = pbest[np.argmax(pbest_val)].copy()
    gbest_val = np.max(pbest_val)

    w, c1, c2 = 0.5, 1.5, 1.5
    for it in range(n_iter):
        r1 = rng.uniform(0, 1, size=(n_particles, n_features))
        r2 = rng.uniform(0, 1, size=(n_particles, n_features))
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        pos = np.clip(pos + vel, 0.0, 1.0)

        for i in range(n_particles):
            mask = pos[i] > 0.5
            if mask.sum() == 0: mask[rng.choice(n_features, 3, replace=False)] = True
            fit = evaluate_subset_cv(mask, X_train, y_train, clf_type)
            if fit > pbest_val[i]:
                pbest_val[i] = fit
                pbest[i] = pos[i].copy()
                if fit > gbest_val:
                    gbest_val = fit
                    gbest = pos[i].copy()

    final_mask = gbest > 0.5
    if final_mask.sum() == 0: final_mask[rng.choice(n_features, 3, replace=False)] = True
    return final_mask

# ----------------------------------------------------------------------------- MAIN LOTO
def main():
    print("=" * 72)
    print("  STAGE 2: BIO-INSPIRED FEATURE SELECTION (GA & PSO) BENCHMARK")
    print("=" * 72)
    t0 = time.time()

    X, taluks, lab, cat, idx_by_taluk = load_data(extra_neg=(-1,))
    D = load_geometry(GPKG_PATH, taluks)
    F_tab = taluk_feature_table(X, idx_by_taluk, taluks)

    n = len(taluks)
    model_names = ["GA_SVM", "PSO_SVM", "GA_LogisticReg"]
    scores = {m: np.full(n, np.nan) for m in model_names}
    n_purged = (D <= BUFFER_KM).sum(1) - 1

    print(f"Loaded: {n} Taluks | 25.0 km Spatially Buffered Leave-One-Taluk-Out")
    print("Running Nested Bio-Inspired Optimization across all 54 folds ...")

    for i in range(n):
        tr_idx = np.where(D[i] > BUFFER_KM)[0]
        y_tr = lab[tr_idx]
        X_tr = F_tab[tr_idx]
        X_te = F_tab[[i]]
        
        if min((y_tr == 0).sum(), (y_tr == 1).sum()) < 2:
            continue

        # 1. Genetic Algorithm + Linear SVM
        ga_mask_svm = genetic_algorithm_select(X_tr, y_tr, clf_type="svm")
        cols_ga_svm = np.where(ga_mask_svm)[0]
        m_ga_svm = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 SVC(kernel="linear", C=0.1, class_weight="balanced", probability=True, random_state=SEED))
        m_ga_svm.fit(X_tr[:, cols_ga_svm], y_tr)
        scores["GA_SVM"][i] = float(m_ga_svm.predict_proba(X_te[:, cols_ga_svm])[0, 1])

        # 2. Particle Swarm Optimization + Linear SVM
        pso_mask_svm = particle_swarm_select(X_tr, y_tr, clf_type="svm")
        cols_pso_svm = np.where(pso_mask_svm)[0]
        m_pso_svm = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  SVC(kernel="linear", C=0.1, class_weight="balanced", probability=True, random_state=SEED))
        m_pso_svm.fit(X_tr[:, cols_pso_svm], y_tr)
        scores["PSO_SVM"][i] = float(m_pso_svm.predict_proba(X_te[:, cols_pso_svm])[0, 1])

        # 3. Genetic Algorithm + Logistic Regression
        ga_mask_lr = genetic_algorithm_select(X_tr, y_tr, clf_type="lr")
        cols_ga_lr = np.where(ga_mask_lr)[0]
        m_ga_lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                LogisticRegression(C=0.1, class_weight="balanced", max_iter=500, random_state=SEED))
        m_ga_lr.fit(X_tr[:, cols_ga_lr], y_tr)
        scores["GA_LogisticReg"][i] = float(m_ga_lr.predict_proba(X_te[:, cols_ga_lr])[0, 1])

        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"  --> Progress: [{i + 1:02d}/{n}] folds completed (elapsed: {time.time() - t0:.1f}s)")

    print("\n" + "=" * 72)
    print(f"{'Bio-Inspired Model':<22} {'ROC-AUC':<9} {'Accuracy':<10} {'Balanced Acc'}")
    print("-" * 72)
    
    summary_results = {}
    for name, s_arr in scores.items():
        auc = roc_auc_score(lab, s_arr)
        pred = (s_arr >= 0.50).astype(int)
        acc = (pred == lab).mean() * 100
        bacc = balanced_accuracy_score(lab, pred) * 100
        print(f"{name:<22} {auc*100:5.1f}%    {acc:5.1f}%     {bacc:5.1f}%")
        summary_results[name] = {"auc": float(auc), "acc": float(acc), "bacc": float(bacc)}
    print("=" * 72)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "stage2_bio_scores.csv")
    pd.DataFrame({"taluk": taluks, "label": lab, "n_purged": n_purged, **scores}).to_csv(out_csv, index=False)
    with open(os.path.join(OUT_DIR, "stage2_summary.json"), "w") as f:
        json.dump(summary_results, f, indent=2)

    print(f"\nSaved Stage 2 results to -> {out_csv}")
    print(f"Total Stage 2 runtime: {time.time() - t0:.2f} seconds.")

if __name__ == "__main__":
    main()
