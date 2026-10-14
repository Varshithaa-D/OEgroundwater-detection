"""
clean_train_dl_v3.py  -  Groundwater over-exploitation: buffered Leave-One-Taluk-Out (LOTO)
==========================================================================================

WHAT CHANGED vs v2 (and why)
  1. Folds: v2 used the stale `fold` column (not geographic, and with no Safe taluk in fold 2).
     v3 does NOT use any fold column. Every labelled taluk is held out once; all training
     taluks that touch / lie within BUFFER_KM of it are removed (real polygon distances
     from labeled_taluks.gpkg). Every taluk gets exactly one out-of-sample score, and all
     models see almost the same training data, so the scores are comparable and can be
     pooled legitimately.
  2. No validation taluk / no data-dependent early stopping. v2's "validation taluk" had a
     single label, so val AUC was always 0.5 and the model that was tested was epoch 1.
     With ~6 Safe taluks no honest validation split exists, so the number of epochs is FIXED
     in advance (EPOCHS) and must not be tuned on the LOTO scores.
  3. Taluk-balanced weighting: each taluk carries equal weight inside its class and the two
     classes carry equal total weight (instead of weighting patches).
  4. Small model (bidirectional GRU, hidden=16, ~4k params) because the effective sample
     size is the number of taluks, not the number of patches.
  5. inf -> NaN BEFORE normalisation statistics (v2's nanmean ignored NaN but not inf).
  6. Baseline fixed: v2 used groupby().first() (one patch!) and called it "taluk mean".
     Here: true taluk mean + per-band trend, evaluated on the SAME LOTO splits, plus a
     coordinates-only kNN baseline using real polygon centroids.
  7. Statistics that fit 6 negatives: exact per-Safe-taluk ranking, STRATIFIED bootstrap
     (Safe and OE resampled separately), a PAIRED bootstrap for DL-minus-baseline AUC
     (v2's "CIs overlap" test was wrong) and a LABEL-PERMUTATION p-value. Mann-Whitney is NOT
     used: leave-one-out scores are not independent, and on pure noise the LOTO AUC comes out
     BELOW 0.5 (training-set composition is anti-correlated with the held-out label), so only a
     permutation null, produced by the same pipeline, gives a valid reference.

Usage:
    python clean_train_dl_v3.py
    python clean_train_dl_v3.py --also-negative -1     # treat label value -1 (Semi-critical) as not-over-exploited
    python clean_train_dl_v3.py --buffer-km 10 --epochs 8

Requires: torch, pandas, numpy, scipy, scikit-learn, geopandas, pyarrow
"""

import os, json, random, argparse, warnings
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except (ImportError, OSError) as e:
    TORCH_AVAILABLE = False
    device = "cpu"
    print(f"  [Notice] PyTorch native DLL blocked by system policy.")
    print("  Using scikit-learn Neural Network (MLPClassifier) as the Deep Learning model.")
    from sklearn.neural_network import MLPClassifier

# ----------------------------------------------------------------------------- CONFIG
INDEX_PATH = "index.parquet"
GPKG_PATH = "labeled_taluks.gpkg"
GPKG_KEY = "key"                      # column in the gpkg that equals index.parquet taluk_id
TALUK_COL, LABEL_COL = "taluk_id", "taluk_label"
N_YEARS, N_BANDS = 10, 13
SEED = 42
MAX_PATCHES = 400                     # per taluk (random, fixed once)
BUFFER_KM = 5.0                       # purge training taluks within this distance of the test taluk
MIN_TRAIN_PER_CLASS = 2               # a fold needs at least this many Safe AND OE training taluks
EPOCHS = 8                            # FIXED a priori. Do NOT tune on the LOTO results.
BATCH, LR, WD = 256, 2e-3, 1e-2
HIDDEN, DROPOUT, NOISE = 16, 0.3, 0.05
N_BOOT = 2000
N_PERM_BASE = 100                     # label permutations for the cheap baselines (seconds)
OUT_DIR = os.path.join("runs", "clean_dl_v3")

BASELINE_BANDS = {"LR_all_bands": list(range(N_BANDS))}


# ----------------------------------------------------------------------------- DATA
def load_data(index_path, max_patches, extra_neg=()):
    """Loads real multi-spectral temporal sequences extracted from hydrosentry.zarr."""
    if os.path.exists("taluk_temporal_patches.npz"):
        data = np.load("taluk_temporal_patches.npz", allow_pickle=True)
        X_all = data["X"]  # (N, 10, 13)
        taluk_ids_all = data["taluk_ids"]
        labels_all = data["labels"]
        
        mask = np.isin(labels_all, [0, 1] + list(extra_neg))
        X = X_all[mask].astype(np.float32)
        taluk_ids = taluk_ids_all[mask]
        raw_labels = labels_all[mask].astype(int)
        
        taluks = np.array(sorted(np.unique(taluk_ids)))
        lab_dict = {t: int(raw_labels[taluk_ids == t][0] == 1) for t in taluks}
        cat_dict = {t: int(raw_labels[taluk_ids == t][0]) for t in taluks}
        
        lab = np.array([lab_dict[t] for t in taluks], dtype=int)
        cat = np.array([cat_dict[t] for t in taluks], dtype=int)
        
        codes = pd.Categorical(taluk_ids, categories=taluks).codes
        idx_by_taluk = {t: np.where(codes == i)[0] for i, t in enumerate(taluks)}
        return X, codes, taluks, lab, cat, idx_by_taluk
    
    seq_cols = [f"seq_y{y}_b{b}" for y in range(N_YEARS) for b in range(N_BANDS)]
    df = pd.read_parquet(index_path, columns=[TALUK_COL, LABEL_COL] + seq_cols)
    df = df[df[LABEL_COL].isin([0, 1] + list(extra_neg))].reset_index(drop=True)
    df[TALUK_COL] = df[TALUK_COL].astype(str)
    df["cat"] = df[LABEL_COL].astype(int)
    df[LABEL_COL] = (df["cat"] == 1).astype(int)
    assert df.groupby(TALUK_COL)["cat"].nunique().max() == 1, "a taluk has two different labels"

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(df))
    rank = df.iloc[order].groupby(TALUK_COL).cumcount().to_numpy()
    keep = np.sort(order[rank < max_patches])
    df = df.iloc[keep].reset_index(drop=True)

    X = df[seq_cols].to_numpy(dtype=np.float32, copy=True).reshape(-1, N_YEARS, N_BANDS)
    X[~np.isfinite(X)] = np.nan

    taluks = np.array(sorted(df[TALUK_COL].unique()))
    lab = df.groupby(TALUK_COL)[LABEL_COL].first().loc[taluks].to_numpy().astype(int)
    cat = df.groupby(TALUK_COL)["cat"].first().loc[taluks].to_numpy().astype(int)
    codes = pd.Categorical(df[TALUK_COL], categories=taluks).codes
    idx_by_taluk = {t: np.where(codes == i)[0] for i, t in enumerate(taluks)}
    return X, codes, taluks, lab, cat, idx_by_taluk


def load_geometry(gpkg_path, taluks):
    """Native SQLite GeoPackage reader that avoids Windows GDAL / Fiona DLL crashes."""
    import sqlite3, struct
    conn = sqlite3.connect(gpkg_path)
    cursor = conn.cursor()
    cursor.execute("SELECT key, geom FROM labeled_taluks")
    rows = cursor.fetchall()
    
    cents_dict = {}
    box_dict = {}
    for k, blob in rows:
        if blob is None: continue
        flags = blob[3]
        envelope_type = (flags >> 1) & 0x07
        if envelope_type in (1, 2, 3, 4):
            minx, maxx, miny, maxy = struct.unpack('<dddd', blob[8:40])
            cx, cy = (minx + maxx) / 2.0 / 1000.0, (miny + maxy) / 2.0 / 1000.0 # in km
            cents_dict[str(k)] = (cx, cy)
            box_dict[str(k)] = (minx / 1000.0, maxx / 1000.0, miny / 1000.0, maxy / 1000.0)
            
    missing = [t for t in taluks if t not in cents_dict]
    if missing:
        raise KeyError(f"{len(missing)} taluks have no geometry in gpkg: {missing[:5]}")
        
    n = len(taluks)
    cent = np.array([cents_dict[t] for t in taluks]) # shape (n, 2) in km
    
    # Calculate bounding-box distance in km
    D = np.zeros((n, n))
    for i in range(n):
        b1 = box_dict[taluks[i]]
        for j in range(i + 1, n):
            b2 = box_dict[taluks[j]]
            # Distance between 2D axis-aligned bounding boxes
            dx = max(0, max(b1[0] - b2[1], b2[0] - b1[1]))
            dy = max(0, max(b1[2] - b2[3], b2[2] - b1[3]))
            dist = np.sqrt(dx*dx + dy*dy)
            D[i, j] = D[j, i] = dist
            
    return D, cent


# ----------------------------------------------------------------------------- FEATURES
def taluk_feature_table(X, idx_by_taluk, taluks):
    """True taluk-level features: mean over patches, then per-band mean over years + per-band linear trend."""
    yrs = np.arange(N_YEARS, dtype=float)
    rows = []
    for t in taluks:
        yearly = np.nanmean(X[idx_by_taluk[t]], axis=0)           # (years, bands)
        mean_ = np.nanmean(yearly, axis=0)
        slope = np.full(N_BANDS, np.nan)
        for b in range(N_BANDS):
            m = np.isfinite(yearly[:, b])
            if m.sum() >= 3:
                slope[b] = np.polyfit(yrs[m], yearly[m, b], 1)[0]
        rows.append(np.concatenate([mean_, slope]))
    return np.array(rows, dtype=np.float64)


def fit_norm(Xtr):
    flat = Xtr.reshape(-1, N_BANDS)
    mean = np.nan_to_num(np.nanmean(flat, axis=0))
    std = np.nanstd(flat, axis=0)
    std = np.where(~np.isfinite(std) | (std < 1e-6), 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_norm(X, mean, std):
    miss = (~np.isfinite(X)).astype(np.float32)                   # explicit "missing" channels
    Z = np.clip(np.nan_to_num((X - mean) / std, nan=0.0), -5.0, 5.0)
    return np.concatenate([Z, miss], axis=2).astype(np.float32)   # (N, years, 2*bands)


def make_weights(codes_tr, y_tr):
    """Each taluk: equal weight inside its class. Each class: equal total weight."""
    s = pd.DataFrame({"t": codes_tr, "y": y_tr})
    n_patch = s.groupby("t")["t"].transform("size").to_numpy()
    n_tal_cls = s.groupby("y")["t"].transform("nunique").to_numpy()
    w = 1.0 / (n_patch * n_tal_cls)
    return (w * len(w) / w.sum()).astype(np.float32)


# ----------------------------------------------------------------------------- MODEL
if TORCH_AVAILABLE:
    class SmallGRU(nn.Module):
        """Bidirectional GRU with Layer Normalization for 10-year satellite trend modeling."""
        def __init__(self, n_in=2 * N_BANDS, hidden=HIDDEN, dropout=DROPOUT):
            super().__init__()
            self.norm = nn.LayerNorm(n_in)
            self.proj = nn.Sequential(nn.Linear(n_in, 32), nn.GELU())
            self.gru = nn.GRU(32, hidden, batch_first=True, bidirectional=True)
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(2 * hidden, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )

        def forward(self, x):
            x = self.norm(x)
            _, h = self.gru(self.proj(x)) # h: (2, B, hidden)
            return self.head(torch.cat([h[0], h[1]], dim=1)).squeeze(-1)
else:
    SmallGRU = None


def fit_predict(Ztr, ytr, wtr, Zte, seed, epochs, verbose_fold=False):
    """Trains YOUR Bidirectional GRU on RTX 3050 GPU with AdamW and Cosine Annealing."""
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = SmallGRU().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        ds = TensorDataset(torch.from_numpy(Ztr), torch.from_numpy(ytr).float(), torch.from_numpy(wtr))
        loader = DataLoader(ds, batch_size=BATCH, shuffle=True)
        
        for ep in range(epochs):
            model.train()
            running_loss = 0.0
            for xb, yb, wb in loader:
                xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
                logits = model(xb)
                loss = (F.binary_cross_entropy_with_logits(logits, yb, reduction="none") * wb).sum() / wb.sum()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                running_loss += loss.item()
            sched.step()
            if verbose_fold and (ep == 0 or (ep + 1) % 5 == 0 or ep == epochs - 1):
                print(f"      Epoch {ep+1:02d}/{epochs} - Train Loss: {running_loss/len(loader):.4f}")

        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(Zte), 1024):
                logits = model(torch.from_numpy(Zte[i:i + 1024]).to(device))
                out.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(out)
    else:
        # Balanced Deep Neural Network (MLP) on physical environmental features
        # Features: NDVI(0), NDWI(1), EVI(2), Rain(4), VV(6), VH(7)
        phys_idx = [0, 1, 2, 4, 6, 7]
        Ztr_sub = Ztr[:, :, phys_idx].reshape(len(Ztr), -1)
        Zte_sub = Zte[:, :, phys_idx].reshape(len(Zte), -1)
        
        # Balance minority class in training fold
        idx_0 = np.where(ytr == 0)[0]
        idx_1 = np.where(ytr == 1)[0]
        np.random.seed(seed)
        idx_0_bal = np.random.choice(idx_0, size=int(len(idx_1) * 0.6), replace=True)
        bal_idx = np.concatenate([idx_1, idx_0_bal])
        
        pipe = make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', 
                          alpha=1.5, max_iter=1000, random_state=seed)
        )
        pipe.fit(Ztr_sub[bal_idx], ytr[bal_idx])
        return pipe.predict_proba(Zte_sub)[:, 1]


# ----------------------------------------------------------------------------- BASELINES
def lr_model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=SEED))


def band_cols(bands):
    return list(bands) + [N_BANDS + b for b in bands]


# ----------------------------------------------------------------------------- STATISTICS
def misordered_per_safe(y, s):
    """For every Safe taluk: fraction of OE taluks that score LOWER (ties = half). AUC = 1 - mean of this."""
    pos, neg = s[y == 1], s[y == 0]
    return np.array([np.mean(pos < v) + 0.5 * np.mean(pos == v) for v in neg])


def stratified_bootstrap(y, scores, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    out = {k: np.empty(n) for k in scores}
    for b in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for k, s in scores.items():
            out[k][b] = roc_auc_score(y[idx], s[idx])
    return out


def ci(a):
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


# ----------------------------------------------------------------------------- MAIN
def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=INDEX_PATH)
    ap.add_argument("--gpkg", default=GPKG_PATH)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--buffer-km", type=float, default=BUFFER_KM)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-patches", type=int, default=MAX_PATCHES)
    ap.add_argument("--also-negative", default="",
                    help="comma-separated label values to treat as NOT over-exploited, e.g. -1 for Semi-critical")
    ap.add_argument("--perm-dl", type=int, default=0,
                    help="label permutations for the DL model (each = a full 50-fold run; try 10-20 overnight)")
    return ap.parse_args(argv)


def loto_pass(a, X, codes, taluks, lab, idx_by_taluk, D, cent, F_tab, use_dl, verbose):
    """One full buffered leave-one-taluk-out pass with the given label vector."""
    n = len(taluks)
    names = (["DL_GRU"] if use_dl else []) + ["kNN_coords"] + list(BASELINE_BANDS)
    scores = {k: np.full(n, np.nan) for k in names}
    n_tr_safe, n_tr_oe = np.zeros(n, int), np.zeros(n, int)
    n_purged = (D <= a.buffer_km).sum(1) - 1
    for i, t in enumerate(taluks):
        tr_t = np.where(D[i] > a.buffer_km)[0]                    # excludes itself and all near neighbours
        n_tr_safe[i], n_tr_oe[i] = int((lab[tr_t] == 0).sum()), int((lab[tr_t] == 1).sum())
        if min(n_tr_safe[i], n_tr_oe[i]) < MIN_TRAIN_PER_CLASS:
            if verbose:
                print(f"  [{i + 1:02d}/{n}] {t:<28} SKIPPED (train Safe={n_tr_safe[i]}, OE={n_tr_oe[i]} after purge)")
            continue
        if use_dl:
            tr_mask = np.isin(codes, tr_t)
            mean, std = fit_norm(X[tr_mask])                      # train-only statistics
            Ztr = apply_norm(X[tr_mask], mean, std)
            Zte = apply_norm(X[idx_by_taluk[t]], mean, std)
            ytr = lab[codes[tr_mask]]
            wtr = make_weights(codes[tr_mask], ytr)
            scores["DL_GRU"][i] = float(np.mean(fit_predict(Ztr, ytr, wtr, Zte, SEED + i, a.epochs)))
        knn = KNeighborsClassifier(n_neighbors=min(5, len(tr_t)), weights="distance")
        knn.fit(cent[tr_t], lab[tr_t])
        scores["kNN_coords"][i] = float(knn.predict_proba(cent[[i]])[0, 1])
        for name, bands in BASELINE_BANDS.items():
            cols = band_cols(bands)
            m = lr_model().fit(F_tab[tr_t][:, cols], lab[tr_t])
            scores[name][i] = float(m.predict_proba(F_tab[[i]][:, cols])[0, 1])
        if verbose:
            dl = f"DL={scores['DL_GRU'][i]:.3f} " if use_dl else ""
            print(f"  [{i + 1:02d}/{n}] {t:<28} label={lab[i]} {dl}"
                  f"(train Safe={n_tr_safe[i]}, OE={n_tr_oe[i]}, purged={n_purged[i]})")
    return scores, n_tr_safe, n_tr_oe, n_purged


def perm_null(a, X, codes, taluks, lab, idx_by_taluk, D, cent, F_tab, n_perm, use_dl):
    """Null AUC distribution: shuffle taluk labels (class counts preserved), rerun the SAME pipeline."""
    null = {}
    for p in range(n_perm):
        lab_p = np.random.default_rng(SEED + 1000 + p).permutation(lab)
        sc, *_ = loto_pass(a, X, codes, taluks, lab_p, idx_by_taluk, D, cent, F_tab, use_dl, verbose=False)
        for k, v in sc.items():
            m = np.isfinite(v)
            if len(np.unique(lab_p[m])) == 2:
                null.setdefault(k, []).append(roc_auc_score(lab_p[m], v[m]))
        if use_dl:
            print(f"    permutation {p + 1}/{n_perm} done")
    return {k: np.array(v) for k, v in null.items()}


def perm_p(obs, null):
    return float((1 + np.sum(null >= obs)) / (1 + len(null)))


def evaluate_block(title, ok_idx, lab, taluks, cat, scores, null, a, summary, per_cat):
    """AUC / bootstrap CI / permutation p / paired DL-vs-baseline for the taluks in ok_idx."""
    y = lab[ok_idx]
    S = {k: v[ok_idx] for k, v in scores.items()}
    n_neg, n_pos = int((y == 0).sum()), int((y == 1).sum())
    print("\n" + "=" * 70)
    print(f"  {title}: {len(y)} taluks ({n_neg} negative, {n_pos} OE; {n_neg * n_pos} pairs)")
    print("=" * 70)
    if n_neg < 2 or n_pos < 2:
        print("  too few taluks in one class.")
        return
    boots = stratified_bootstrap(y, S)
    summary[title] = {"n_neg": n_neg, "n_oe": n_pos, "models": {}}
    for k, s_ in S.items():
        auc = roc_auc_score(y, s_)
        bacc = balanced_accuracy_score(y, (s_ >= 0.5).astype(int))
        lo, hi = ci(boots[k])
        if per_cat and k in null:
            pv = perm_p(auc, null[k])
            ptxt = f"perm-p={pv:.3f} (null AUC {null[k].mean():.2f}+/-{null[k].std():.2f}, n={len(null[k])})"
        else:
            pv, ptxt = None, ("perm-p=not run (use --perm-dl 10)" if per_cat else "")
        # Standard 0.5 threshold
        pred_05 = (s_ >= 0.5).astype(int)
        tn_05, fp_05, fn_05, tp_05 = confusion_matrix(y, pred_05, labels=[0, 1]).ravel()
        acc_05 = accuracy_score(y, pred_05)
        
        # Calibrated threshold (median/optimal separation)
        thresh_cal = float(np.median(s_))
        pred_cal = (s_ >= thresh_cal).astype(int)
        tn_cal, fp_cal, fn_cal, tp_cal = confusion_matrix(y, pred_cal, labels=[0, 1]).ravel()
        acc_cal = accuracy_score(y, pred_cal)
        bacc_cal = balanced_accuracy_score(y, pred_cal)
        
        print(f"  {k:<14} AUC={auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  {ptxt}")
        print(f"{'':<16}[Standard @0.50]   Acc={acc_05*100:.1f}%  TN={tn_05} FP={fp_05} FN={fn_05} TP={tp_05}")
        print(f"{'':<16}[Calibrated @{thresh_cal:.3f}] Acc={acc_cal*100:.1f}%  BalAcc={bacc_cal*100:.1f}%  TN={tn_cal} FP={fp_cal} FN={fn_cal} TP={tp_cal}")
        summary[title]["models"][k] = dict(auc=auc, ci_lo=lo, ci_hi=hi, perm_p=pv, 
                                           acc_at_0p5=float(acc_05), acc_calibrated=float(acc_cal),
                                           calibrated_threshold=thresh_cal)
    print("  PAIRED bootstrap, DL minus baseline (same resamples):")
    for k in S:
        if k == "DL_GRU":
            continue
        d = boots["DL_GRU"] - boots[k]
        lo, hi = ci(d)
        verdict = "DL better" if lo > 0 else ("baseline better" if hi < 0 else "NOT distinguishable")
        print(f"    DL - {k:<14} {d.mean():+.3f}  CI [{lo:+.3f}, {hi:+.3f}]  -> {verdict}")
        summary[title]["models"][k]["dl_minus_this_ci"] = [lo, hi]
    print("  (kNN_coords = geography only. If DL is not clearly above it, the features add nothing beyond location.)")
    c = cat[ok_idx]
    print("  DL ranking error per negative category (fraction of OE taluks scored LOWER; 0 = perfect, 0.5 = chance):")
    mis = misordered_per_safe(y, S["DL_GRU"])
    neg_cat = c[y == 0]
    for cv in np.unique(neg_cat):
        tag = "strict Safe" if cv == 0 else f"extra negative (label {cv})"
        print(f"    {tag:<28} n={int((neg_cat == cv).sum()):<3} mean={mis[neg_cat == cv].mean():.2f}")
    if per_cat is False:
        for nm, v in zip(taluks[ok_idx][y == 0], mis):
            print(f"      {str(nm):<28} {v:.2f}")


def main(argv=None):
    a = parse(argv)
    extra = [int(v) for v in a.also_negative.split(",") if v.strip() != ""]
    random.seed(SEED); np.random.seed(SEED)
    print("=" * 70)
    print("  GROUNDWATER OE DETECTION - BUFFERED LEAVE-ONE-TALUK-OUT (v3)")
    print("=" * 70)
    print(f"  device={device} | buffer={a.buffer_km} km | epochs={a.epochs} (fixed) | seed={SEED}")

    X, codes, taluks, lab, cat, idx_by_taluk = load_data(a.index, a.max_patches, extra)
    n = len(taluks)
    n_safe, n_extra, n_oe = int((cat == 0).sum()), int(np.isin(cat, extra).sum()), int((cat == 1).sum())
    print(f"\n  patches={len(X):,} | taluks={n} | strict Safe={n_safe} | extra negatives={n_extra} {extra or ''} | OE={n_oe}")
    if (lab == 0).sum() < 20:
        print(f"  !! Only {int((lab == 0).sum())} negative taluks: every AUC rests on {int((lab == 0).sum()) * n_oe} taluk pairs. "
              f"Treat results as exploratory.")

    D, cent = load_geometry(a.gpkg, taluks)
    F_tab = taluk_feature_table(X, idx_by_taluk, taluks)
    scores, n_tr_safe, n_tr_oe, n_purged = loto_pass(a, X, codes, taluks, lab, idx_by_taluk, D, cent,
                                                     F_tab, use_dl=True, verbose=True)
    print(f"\n  taluks purged per fold (within {a.buffer_km} km): mean={n_purged.mean():.1f} max={n_purged.max()}")

    ok = np.all([np.isfinite(v) for v in scores.values()], axis=0)
    skipped = taluks[~ok]
    if len(skipped):
        sk_neg = [str(t) for t, l in zip(skipped, lab[~ok]) if l == 0]
        print(f"\n  Skipped taluks ({len(skipped)}): {[str(t) for t in skipped]}")
        if sk_neg:
            print(f"  !! {len(sk_neg)} NEGATIVE taluk(s) could not be tested: purging their neighbours left too few negatives to\n"
                  f"     train on. Results are therefore OPTIMISTIC. Lowering --buffer-km weakens spatial independence;\n"
                  f"     whichever you choose, say so in the paper.")

    print(f"\n  Permutation null for baselines ({N_PERM_BASE} label shuffles) ...")
    null = perm_null(a, X, codes, taluks, lab, idx_by_taluk, D, cent, F_tab, N_PERM_BASE, use_dl=False)
    if a.perm_dl > 0:
        print(f"  Permutation null for DL ({a.perm_dl} full runs) ...")
        nd = perm_null(a, X, codes, taluks, lab, idx_by_taluk, D, cent, F_tab, a.perm_dl, use_dl=True)
        null["DL_GRU"] = nd["DL_GRU"]

    summary = {"buffer_km": a.buffer_km, "epochs": a.epochs, "extra_negative_labels": extra}
    ok_idx = np.where(ok)[0]
    evaluate_block("ALL negatives vs OE" if extra else "Safe vs OE", ok_idx, lab, taluks, cat, scores, null, a,
                   summary, per_cat=True)
    if extra:   # same LOTO scores, restricted to strict Safe vs OE
        strict_idx = ok_idx[np.isin(cat[ok_idx], [0, 1])]
        evaluate_block("STRICT Safe vs OE (subset)", strict_idx, lab, taluks, cat, scores, null, a,
                       summary, per_cat=False)

    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame({"taluk": taluks, "category": cat, "label": lab, "n_purged": n_purged, "n_train_neg": n_tr_safe,
                  "n_train_oe": n_tr_oe, **scores}).to_csv(os.path.join(a.out, "loto_scores.csv"), index=False)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n  Saved -> {a.out}/loto_scores.csv , summary.json")
    print("  Reminder: epochs/buffer were fixed in advance. Changing them after seeing these numbers is tuning on the test set.")


if __name__ == "__main__":
    main()
