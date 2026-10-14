"""
clean_train_dl_v4.py - Groundwater over-exploitation: buffered Leave-One-Taluk-Out (LOTO)
=========================================================================================

Fixes vs v3
  1. No test-derived threshold. Both models are trained with class-balanced weights, so 0.5
     is the pre-specified balanced threshold. Headline metrics are AUC and balanced accuracy.
  2. Buffer default 25 km. Distances are BOUNDING-BOX distances; CRS is read from the gpkg,
     geographic coordinates are converted to km, and a taluk-size sanity check is printed.
  3. Missing envelope / wrong CRS raises a clear error.
  4. Stops cleanly if PyTorch native DLL cannot load.
  5. LR / WD constants used; Gaussian NOISE regularisation active during training.
  6. Uses taluk_temporal_patches.npz cache (real multi-spectral data).
  7. Multi-seed averaging (--seeds, default 1 or 3).
  8. Permutation test for DL (--perm-dl, default 0 for speed, 10 for paper).
"""

import os, json, random, argparse, warnings, time
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception as _torch_err:
    torch = None
    nn = None
    F = None
    TensorDataset = None
    DataLoader = None
    device = "cpu"
    warnings.warn(f"PyTorch could not be loaded ({_torch_err}). Deep learning routines will be unavailable.")

# ----------------------------------------------------------------------------- CONFIG
INDEX_PATH = "index.parquet"
GPKG_PATH = "labeled_taluks.gpkg"
GPKG_TABLE, GPKG_KEY = "labeled_taluks", "key"
NPZ_PATH = "taluk_temporal_patches.npz"
TALUK_COL, LABEL_COL = "taluk_id", "taluk_label"
N_YEARS, N_BANDS = 10, 13
SEED = 42
MAX_PATCHES = 400
BUFFER_KM = 25.0
MIN_TRAIN_PER_CLASS = 2
EPOCHS = 15
BATCH, LR, WD = 256, 1e-3, 1e-2       # AdamW
HIDDEN, DROPOUT, NOISE = 16, 0.3, 0.05
THRESH = 0.5
N_BOOT = 2000
N_PERM_BASE = 100
REF_LAT = 15.0
OUT_DIR = os.path.join("runs", "clean_dl_v4")

BASELINE_BANDS = {"LR_all_bands": list(range(N_BANDS))}


# ----------------------------------------------------------------------------- DATA
def _finish(X, taluk_ids, raw_labels):
    taluks = np.array(sorted(np.unique(taluk_ids)))
    lab = np.array([int(raw_labels[taluk_ids == t][0] == 1) for t in taluks])
    cat = np.array([int(raw_labels[taluk_ids == t][0]) for t in taluks])
    codes = pd.Categorical(taluk_ids, categories=taluks).codes
    idx_by_taluk = {t: np.where(codes == i)[0] for i, t in enumerate(taluks)}
    return X, codes, taluks, lab, cat, idx_by_taluk


def load_data(index_path, max_patches, extra_neg=(), use_npz=True):
    if use_npz and os.path.exists(NPZ_PATH):
        st = os.stat(NPZ_PATH)
        print(f"  !! Using CACHE {NPZ_PATH} (modified {time.ctime(st.st_mtime)}).")
        d = np.load(NPZ_PATH, allow_pickle=True)
        mask = np.isin(d["labels"], [0, 1] + list(extra_neg))
        X = d["X"][mask].astype(np.float32)
        X[~np.isfinite(X)] = np.nan
        print(f"  cache shape: {X.shape}")
        return _finish(X, d["taluk_ids"][mask].astype(str), d["labels"][mask].astype(int))

    print(f"  Reading sequences from {index_path}")
    seq_cols = [f"seq_y{y}_b{b}" for y in range(N_YEARS) for b in range(N_BANDS)]
    df = pd.read_parquet(index_path, columns=[TALUK_COL, LABEL_COL] + seq_cols)
    df = df[df[LABEL_COL].isin([0, 1] + list(extra_neg))].reset_index(drop=True)
    df[TALUK_COL] = df[TALUK_COL].astype(str)
    assert df.groupby(TALUK_COL)[LABEL_COL].nunique().max() == 1, "a taluk has two different labels"
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(df))
    rank = df.iloc[order].groupby(TALUK_COL).cumcount().to_numpy()
    df = df.iloc[np.sort(order[rank < max_patches])].reset_index(drop=True)
    X = df[seq_cols].to_numpy(dtype=np.float32, copy=True).reshape(-1, N_YEARS, N_BANDS)
    X[~np.isfinite(X)] = np.nan
    return _finish(X, df[TALUK_COL].to_numpy(), df[LABEL_COL].to_numpy().astype(int))


def load_geometry(gpkg_path, taluks):
    import sqlite3, struct
    con = sqlite3.connect(gpkg_path)
    cur = con.cursor()
    cur.execute("SELECT column_name, srs_id FROM gpkg_geometry_columns WHERE table_name=?", (GPKG_TABLE,))
    geom_col, srs = cur.fetchone()
    cur.execute(f'SELECT "{GPKG_KEY}", "{geom_col}" FROM "{GPKG_TABLE}"')
    boxes = {}
    for k, blob in cur.fetchall():
        if blob is None or blob[:2] != b"GP":
            continue
        flags = blob[3]
        if ((flags >> 1) & 7) == 0:
            raise ValueError("GeoPackage geometry has no envelope; re-export with envelopes.")
        minx, maxx, miny, maxy = struct.unpack("<dddd" if flags & 1 else ">dddd", blob[8:40])
        boxes[str(k)] = (minx, maxx, miny, maxy)
    con.close()

    missing = [t for t in taluks if t not in boxes]
    if missing:
        raise KeyError(f"{len(missing)} taluks have no geometry in gpkg: {missing[:5]}")
    B = np.array([boxes[t] for t in taluks], dtype=float)
    if srs in (4326, 4269, 4258):
        kx, ky = 111.32 * np.cos(np.radians(REF_LAT)), 110.57
        B = B * np.array([kx, kx, ky, ky])
        print(f"  gpkg CRS EPSG:{srs} (degrees) -> converted to approx km")
    else:
        B = B / 1000.0
        print(f"  gpkg CRS EPSG:{srs} assumed metric (metres -> km)")
    w = np.median(B[:, 1] - B[:, 0]); h = np.median(B[:, 3] - B[:, 2])
    print(f"  median taluk bbox size: {w:.0f} x {h:.0f} km (Karnataka taluks typically ~20-60 km)")
    if not (5 <= w <= 200 and 5 <= h <= 200):
        raise ValueError("Taluk sizes look wrong - check gpkg CRS/units.")
    cent = np.stack([(B[:, 0] + B[:, 1]) / 2, (B[:, 2] + B[:, 3]) / 2], axis=1)
    dx = np.maximum(0, np.maximum(B[:, None, 0] - B[None, :, 1], B[None, :, 0] - B[:, None, 1]))
    dy = np.maximum(0, np.maximum(B[:, None, 2] - B[None, :, 3], B[None, :, 2] - B[:, None, 3]))
    return np.sqrt(dx ** 2 + dy ** 2), cent


# ----------------------------------------------------------------------------- FEATURES
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


def fit_norm(Xtr):
    flat = Xtr.reshape(-1, N_BANDS)
    mean = np.nan_to_num(np.nanmean(flat, axis=0))
    std = np.nanstd(flat, axis=0)
    std = np.where(~np.isfinite(std) | (std < 1e-6), 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_norm(X, mean, std):
    miss = (~np.isfinite(X)).astype(np.float32)
    Z = np.clip(np.nan_to_num((X - mean) / std, nan=0.0), -5.0, 5.0)
    return np.concatenate([Z, miss], axis=2).astype(np.float32)


def make_weights(codes_tr, y_tr):
    s = pd.DataFrame({"t": codes_tr, "y": y_tr})
    n_patch = s.groupby("t")["t"].transform("size").to_numpy()
    n_tal_cls = s.groupby("y")["t"].transform("nunique").to_numpy()
    w = 1.0 / (n_patch * n_tal_cls)
    return (w * len(w) / w.sum()).astype(np.float32)


# ----------------------------------------------------------------------------- MODEL
class SmallGRU(nn.Module):
    """LayerNorm -> Linear(26,32)+GELU -> bidirectional GRU(hidden 16) -> MLP head. 10-year sequences."""
    def __init__(self, n_in=2 * N_BANDS, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.norm = nn.LayerNorm(n_in)
        self.proj = nn.Sequential(nn.Linear(n_in, 32), nn.GELU())
        self.gru = nn.GRU(32, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        _, h = self.gru(self.proj(self.norm(x)))
        return self.head(torch.cat([h[0], h[1]], dim=1)).squeeze(-1)


def _train_one(Ztr, ytr, wtr, Zte, seed, epochs, verbose_epochs=False):
    torch.manual_seed(seed); np.random.seed(seed)
    model = SmallGRU().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(TensorDataset(torch.from_numpy(Ztr), torch.from_numpy(ytr).float(), torch.from_numpy(wtr)),
                        batch_size=BATCH, shuffle=True)
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            xb = xb.clone()
            xb[..., :N_BANDS] += NOISE * torch.randn_like(xb[..., :N_BANDS])
            loss = (F.binary_cross_entropy_with_logits(model(xb), yb, reduction="none") * wb).sum() / wb.sum()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        sched.step()
        if verbose_epochs and (ep == 0 or (ep + 1) % 5 == 0 or ep == epochs - 1):
            print(f"      Epoch {ep+1:02d}/{epochs} - Train Loss: {total_loss/len(loader):.4f}")
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Zte), 1024):
            out.append(torch.sigmoid(model(torch.from_numpy(Zte[i:i + 1024]).to(device))).cpu().numpy())
    return np.concatenate(out)


def fit_predict(Ztr, ytr, wtr, Zte, seed, epochs, n_seeds, verbose_epochs=False):
    return np.mean([_train_one(Ztr, ytr, wtr, Zte, seed + 7919 * s, epochs, verbose_epochs=(verbose_epochs and s == 0)) for s in range(n_seeds)], axis=0)


# ----------------------------------------------------------------------------- BASELINES
def lr_model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=SEED))


def band_cols(bands):
    return list(bands) + [N_BANDS + b for b in bands]


# ----------------------------------------------------------------------------- STATISTICS
def misordered_per_safe(y, s):
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
    ap.add_argument("--seeds", type=int, default=1, help="GRU seeds averaged per held-out taluk (default 1 for speed, 3 for ensemble)")
    ap.add_argument("--also-negative", default="", help="label values treated as NOT over-exploited, e.g. -1")
    ap.add_argument("--perm-dl", type=int, default=0, help="label permutations for the GRU (0 = skip)")
    ap.add_argument("--no-npz", action="store_true", help="ignore cache npz")
    try:
        ap.add_argument("--show-epochs", action=argparse.BooleanOptionalAction, default=True, help="show epoch loss progression (use --no-show-epochs to disable)")
    except AttributeError:
        ap.add_argument("--show-epochs", action="store_true", default=True, dest="show_epochs", help="show epoch loss progression")
        ap.add_argument("--no-show-epochs", action="store_false", dest="show_epochs", help="suppress epoch loss progression")
    return ap.parse_args(argv)


def loto_pass(a, X, codes, taluks, lab, idx_by_taluk, D, cent, F_tab, use_dl, verbose, n_seeds=1):
    n = len(taluks)
    names = (["DL_GRU"] if use_dl else []) + ["kNN_coords"] + list(BASELINE_BANDS)
    scores = {k: np.full(n, np.nan) for k in names}
    n_tr_safe, n_tr_oe = np.zeros(n, int), np.zeros(n, int)
    n_purged = (D <= a.buffer_km).sum(1) - 1
    for i, t in enumerate(taluks):
        tr_t = np.where(D[i] > a.buffer_km)[0]
        n_tr_safe[i], n_tr_oe[i] = int((lab[tr_t] == 0).sum()), int((lab[tr_t] == 1).sum())
        if min(n_tr_safe[i], n_tr_oe[i]) < MIN_TRAIN_PER_CLASS:
            if verbose:
                print(f"  [{i + 1:02d}/{n}] {t:<28} SKIPPED (train Safe={n_tr_safe[i]}, OE={n_tr_oe[i]} after purge)")
            continue
        if verbose:
            print(f"  [{i + 1:02d}/{n}] {t:<28} label={lab[i]} (train Safe={n_tr_safe[i]}, OE={n_tr_oe[i]}, purged={n_purged[i]})")
        if use_dl:
            tr_mask = np.isin(codes, tr_t)
            mean, std = fit_norm(X[tr_mask])
            Ztr = apply_norm(X[tr_mask], mean, std)
            Zte = apply_norm(X[idx_by_taluk[t]], mean, std)
            ytr = lab[codes[tr_mask]]
            wtr = make_weights(codes[tr_mask], ytr)
            scores["DL_GRU"][i] = float(np.mean(fit_predict(Ztr, ytr, wtr, Zte, SEED + i, a.epochs, n_seeds, verbose_epochs=getattr(a, 'show_epochs', True))))
        knn = KNeighborsClassifier(n_neighbors=min(5, len(tr_t)), weights="distance")
        knn.fit(cent[tr_t], lab[tr_t])
        scores["kNN_coords"][i] = float(knn.predict_proba(cent[[i]])[0, 1])
        for name, bands in BASELINE_BANDS.items():
            cols = band_cols(bands)
            m = lr_model().fit(F_tab[tr_t][:, cols], lab[tr_t])
            scores[name][i] = float(m.predict_proba(F_tab[[i]][:, cols])[0, 1])
        if verbose:
            dl = f"DL={scores['DL_GRU'][i]:.3f}" if use_dl else ""
            print(f"    --> Score: {dl} | kNN={scores['kNN_coords'][i]:.3f} | LR={scores['LR_all_bands'][i]:.3f}")
    return scores, n_tr_safe, n_tr_oe, n_purged


def perm_null(a, X, codes, taluks, lab, idx_by_taluk, D, cent, F_tab, n_perm, use_dl):
    null = {}
    for p in range(n_perm):
        lab_p = np.random.default_rng(SEED + 1000 + p).permutation(lab)
        sc, *_ = loto_pass(a, X, codes, taluks, lab_p, idx_by_taluk, D, cent, F_tab, use_dl, verbose=False, n_seeds=1)
        for k, v in sc.items():
            m = np.isfinite(v)
            if len(np.unique(lab_p[m])) == 2:
                null.setdefault(k, []).append(roc_auc_score(lab_p[m], v[m]))
        if use_dl:
            print(f"    permutation {p + 1}/{n_perm} done")
    return {k: np.array(v) for k, v in null.items()}


def perm_p(obs, null):
    return float((1 + np.sum(null >= obs)) / (1 + len(null)))


def evaluate_block(title, ok_idx, lab, taluks, cat, scores, null, summary, per_cat):
    y = lab[ok_idx]
    S = {k: v[ok_idx] for k, v in scores.items()}
    n_neg, n_pos = int((y == 0).sum()), int((y == 1).sum())
    print("\n" + "=" * 70)
    print(f"  {title}: {len(y)} taluks ({n_neg} negative, {n_pos} OE; {n_neg * n_pos} pairs)")
    print(f"  Majority-class (always OE) baseline = {n_pos / len(y) * 100:.1f}%")
    print("=" * 70)
    if n_neg < 2 or n_pos < 2:
        print("  too few taluks in one class.")
        return
    boots = stratified_bootstrap(y, S)
    summary[title] = {"n_neg": n_neg, "n_oe": n_pos, "majority_acc": n_pos / len(y), "models": {}}
    for k, s_ in S.items():
        auc = roc_auc_score(y, s_)
        lo, hi = ci(boots[k])
        if per_cat and k in null:
            pv = perm_p(auc, null[k])
            ptxt = f"perm-p={pv:.3f} (null AUC {null[k].mean():.2f}+/-{null[k].std():.2f}, n={len(null[k])})"
        else:
            pv, ptxt = None, ("perm-p=not run" if per_cat else "")
        pred = (s_ >= THRESH).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        bacc = balanced_accuracy_score(y, pred)
        acc = (pred == y).mean()
        print(f"  {k:<14} AUC={auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  {ptxt}")
        print(f"{'':<16}@{THRESH:.2f}: Acc={acc*100:.1f}%  BalAcc={bacc * 100:.1f}%  (TN={tn} FP={fp} FN={fn} TP={tp})")
        summary[title]["models"][k] = dict(auc=auc, ci_lo=lo, ci_hi=hi, perm_p=pv, acc_at_0p5=float(acc), bal_acc_at_0p5=float(bacc))
    if "DL_GRU" in S:
        print("  PAIRED bootstrap, DL minus baseline (same resamples):")
        for k in S:
            if k == "DL_GRU":
                continue
            d = boots["DL_GRU"] - boots[k]
            lo, hi = ci(d)
            verdict = "DL better" if lo > 0 else ("baseline better" if hi < 0 else "NOT distinguishable")
            print(f"    DL - {k:<14} {d.mean():+.3f}  CI [{lo:+.3f}, {hi:+.3f}]  -> {verdict}")
            summary[title]["models"][k]["dl_minus_this_ci"] = [lo, hi]
        mis = misordered_per_safe(y, S["DL_GRU"])
        neg_cat = cat[ok_idx][y == 0]
        print("  DL ranking error per negative category (fraction of OE scored LOWER; 0 perfect, 0.5 chance):")
        for cv in np.unique(neg_cat):
            tag = "strict Safe" if cv == 0 else f"extra negative (label {cv})"
            print(f"    {tag:<28} n={int((neg_cat == cv).sum()):<3} mean={mis[neg_cat == cv].mean():.2f}")
        if per_cat is False:
            for nm, v in zip(taluks[ok_idx][y == 0], mis):
                print(f"      {str(nm):<28} {v:.2f}")


def main(argv=None):
    a = parse(argv)
    extra = [int(v) for v in a.also_negative.split(",") if v.strip() != ""]
    if a.lr_subset.strip():
        BASELINE_BANDS["LR_subset"] = [int(v) for v in a.lr_subset.split(",")]
    random.seed(SEED); np.random.seed(SEED)
    print("=" * 70)
    print("  GROUNDWATER OE DETECTION - BUFFERED LEAVE-ONE-TALUK-OUT (v4)")
    print("=" * 70)
    print(f"  device={device} | buffer={a.buffer_km} km | epochs={a.epochs} (fixed) | seeds={a.seeds} | "
          f"AdamW lr={LR} wd={WD} | seed={SEED}")

    X, codes, taluks, lab, cat, idx_by_taluk = load_data(a.index, a.max_patches, extra, use_npz=not a.no_npz)
    n = len(taluks)
    print(f"\n  patches={len(X):,} | taluks={n} | strict Safe={int((cat == 0).sum())} | "
          f"extra negatives={int(np.isin(cat, extra).sum())} {extra or ''} | OE={int((cat == 1).sum())}")
    if (lab == 0).sum() < 20:
        print(f"  !! Only {int((lab == 0).sum())} negative taluks: every AUC rests on "
              f"{int((lab == 0).sum()) * int((lab == 1).sum())} taluk pairs. Treat results as exploratory.")

    D, cent = load_geometry(a.gpkg, taluks)
    F_tab = taluk_feature_table(X, idx_by_taluk, taluks)
    scores, n_tr_safe, n_tr_oe, n_purged = loto_pass(a, X, codes, taluks, lab, idx_by_taluk, D, cent,
                                                     F_tab, use_dl=True, verbose=True, n_seeds=a.seeds)
    print(f"\n  taluks purged per fold (within {a.buffer_km} km): mean={n_purged.mean():.1f} max={n_purged.max()}")

    ok = np.all([np.isfinite(v) for v in scores.values()], axis=0)
    skipped = taluks[~ok]
    if len(skipped):
        print(f"\n  Skipped taluks ({len(skipped)}): {[str(t) for t in skipped]}")

    print(f"\n  Permutation null for baselines ({N_PERM_BASE} label shuffles) ...")
    null = perm_null(a, X, codes, taluks, lab, idx_by_taluk, D, cent, F_tab, N_PERM_BASE, use_dl=False)
    if a.perm_dl > 0:
        print(f"  Permutation null for DL ({a.perm_dl} full runs, 1 seed each) ...")
        nd = perm_null(a, X, codes, taluks, lab, idx_by_taluk, D, cent, F_tab, a.perm_dl, use_dl=True)
        if "DL_GRU" in nd:
            null["DL_GRU"] = nd["DL_GRU"]

    summary = {"buffer_km": a.buffer_km, "epochs": a.epochs, "seeds": a.seeds, "lr": LR, "wd": WD,
               "extra_negative_labels": extra}
    ok_idx = np.where(ok)[0]
    evaluate_block("ALL negatives vs OE" if extra else "Safe vs OE", ok_idx, lab, taluks, cat, scores, null,
                   summary, per_cat=True)
    if extra:
        strict_idx = ok_idx[np.isin(cat[ok_idx], [0, 1])]
        evaluate_block("STRICT Safe vs OE (subset)", strict_idx, lab, taluks, cat, scores, null, summary, per_cat=False)

    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame({"taluk": taluks, "category": cat, "label": lab, "n_purged": n_purged, "n_train_neg": n_tr_safe,
                  "n_train_oe": n_tr_oe, **scores}).to_csv(os.path.join(a.out, "loto_scores.csv"), index=False)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n  Saved -> {a.out}/loto_scores.csv , summary.json")


if __name__ == "__main__":
    main()
