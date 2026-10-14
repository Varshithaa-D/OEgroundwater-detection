"""
clean_train_unified_v4.py - classical benchmark + ablations for taluk-level over-exploitation (NO deep learning here)
===================================================================================================================
Corrected version of the "unified v3" script. What was wrong there and what this version does instead:

  1. GEOMETRY: v3 divided GPKG envelope coordinates by 1000 without checking the CRS (wrong if the layer is in degrees)
     and silently gave taluks with missing polygons the SAME fake location. Here: the CRS is read from the GPKG,
     degrees are converted to km, missing taluks raise an error (with name suggestions), and, if shapely is
     installed, true POLYGON distances are used (bounding boxes only as a flagged fallback).
  2. CV DESIGN: 5-fold block CV leaves folds with 0-1 Safe taluks and pools five different models. With ~6 Safe taluks
     this version uses buffered LEAVE-ONE-TALUK-OUT: every taluk is scored once by a model that never saw it or its
     neighbours (within --buffer-km). Same protocol as clean_train_dl_v3.py, so scores are directly comparable.
  3. FEATURES: v3 used only 10-year means ("dynamic signal" was never tested). Now: per-band mean + per-band trend.
     inf -> NaN first (inf crashed StandardScaler).
  4. BAND NAMES: the names -> seq_b0..b12 mapping was assumed. It is now CHECKED (static bands must be topography).
  5. EVALUATION: no more nan -> 0.5 filling (silently diluted AUC). Only taluks scored by every model are compared;
     skipped taluks are listed. Reports AUC + stratified-bootstrap CI + paired DL/LR-vs-kNN_coords differences +
     accuracy next to the always-OE baseline + confusion counts. "+/- across repeats" removed (it was not uncertainty).
  6. PERMUTATION test is run through the SAME LOTO pipeline. Caveat printed: labels are spatially clustered, so it only
     tests "better than chance", NOT "better than geography". The kNN_coords comparison is the real control.
  7. --also-negative (e.g. -1 = Semi-critical) and --dl-scores (merge clean_train_dl_v3.py output) added.

Usage:
    python clean_train_unified_v4.py
    python clean_train_unified_v4.py --also-negative -1 --buffer-km 5
    python clean_train_unified_v4.py --dl-scores runs/clean_dl_v3/loto_scores.csv
Requires: numpy pandas scipy scikit-learn pyarrow   (shapely optional but recommended; no torch, no GDAL)
"""
import os, json, argparse, sqlite3, struct, warnings, difflib
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
try:
    from shapely import wkb as _wkb
    from shapely.ops import transform as _transform
    HAVE_SHAPELY = True
except Exception:
    HAVE_SHAPELY = False

SEED = 42
N_BANDS, N_YEARS = 13, 10
MIN_TRAIN_PER_CLASS = 2
OUT_DIR = os.path.join("runs", "unified_v4")


# ----------------------------------------------------------------------------- GEOMETRY (sqlite, no GDAL)
_ENV_LEN = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}


def _split_blob(blob):
    """GeoPackage binary header -> (envelope(minx,miny,maxx,maxy) or None, WKB bytes or None)."""
    if blob is None or len(blob) < 8 or blob[:2] != b"GP":
        return None, None
    flags = blob[3]
    if (flags >> 4) & 1:                                   # empty geometry flag
        return None, None
    endian = "<" if (flags & 1) else ">"
    env_len = _ENV_LEN.get((flags >> 1) & 7)
    if env_len is None:
        return None, None
    env = None
    if env_len:
        minx, maxx, miny, maxy = struct.unpack(endian + "dddd", blob[8:40])
        env = (minx, miny, maxx, maxy)
    return env, blob[8 + env_len:]


def _units(crs_def, envs):
    d = (crs_def or "").upper()
    if "PROJCS" in d or "PROJCRS" in d:                    # check first: WKT1 PROJCS contains a nested GEOGCS
        if "METRE" in d or "METER" in d:
            return "m"
        raise ValueError("Projected CRS is not in metres; reproject labeled_taluks.gpkg to a metre CRS (e.g. EPSG:32643).")
    if "GEOGCS" in d or "GEOGCRS" in d or "GEODCRS" in d:
        return "deg"
    ex = np.array([[e[0], e[2]] for e in envs]).ravel()    # CRS undefined -> guess from magnitude
    ey = np.array([[e[1], e[3]] for e in envs]).ravel()
    return "deg" if (np.abs(ex).max() <= 180 and np.abs(ey).max() <= 90) else "m"


def load_geometry(gpkg, taluks, key_col="key"):
    con = sqlite3.connect(gpkg)
    try:
        tbl, gcol, srs = con.execute("SELECT table_name, column_name, srs_id FROM gpkg_geometry_columns").fetchone()
        row = con.execute("SELECT definition FROM gpkg_spatial_ref_sys WHERE srs_id=?", (srs,)).fetchone()
        crs_def = row[0] if row else ""
        rows = con.execute(f'SELECT "{key_col}", "{gcol}" FROM "{tbl}"').fetchall()
    finally:
        con.close()
    rec = {}
    for k, blob in rows:
        env, wkb_bytes = _split_blob(blob)
        geom = _wkb.loads(wkb_bytes) if (HAVE_SHAPELY and wkb_bytes) else None
        if env is None and geom is not None:
            env = geom.bounds
        if env is not None:
            rec[str(k)] = (env, geom)
    missing = [t for t in taluks if t not in rec]
    if missing:
        hints = {m: difflib.get_close_matches(m, list(rec), n=2) for m in missing[:5]}
        raise KeyError(f"{len(missing)} taluks have no polygon in the gpkg (key column '{key_col}'). Examples/closest names: {hints}")
    units = _units(crs_def, [rec[t][0] for t in taluks])
    if units == "m":
        kx = ky = 1.0 / 1000.0
    else:                                                  # degrees -> km (equirectangular around mean latitude)
        lat0 = np.mean([(rec[t][0][1] + rec[t][0][3]) / 2 for t in taluks])
        kx, ky = 111.32 * np.cos(np.radians(lat0)), 110.57
    B = np.array([[rec[t][0][0] * kx, rec[t][0][2] * kx, rec[t][0][1] * ky, rec[t][0][3] * ky] for t in taluks])
    n = len(taluks)
    if HAVE_SHAPELY and all(rec[t][1] is not None for t in taluks):
        geoms = [_transform(lambda x, y, z=None: (x * kx, y * ky), rec[t][1]) for t in taluks]
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = geoms[i].distance(geoms[j])
        cent = np.array([[g.centroid.x, g.centroid.y] for g in geoms])
        how = "true polygon distance"
    else:
        dx = np.maximum(0, np.maximum(B[:, None, 0] - B[None, :, 1], B[None, :, 0] - B[:, None, 1]))
        dy = np.maximum(0, np.maximum(B[:, None, 2] - B[None, :, 3], B[None, :, 2] - B[:, None, 3]))
        D = np.hypot(dx, dy)
        cent = np.c_[(B[:, 0] + B[:, 1]) / 2, (B[:, 2] + B[:, 3]) / 2]
        how = "BOUNDING-BOX distance (shapely missing/blocked): purges MORE than polygon distance would"
    side = np.median(np.sqrt((B[:, 1] - B[:, 0]) * (B[:, 3] - B[:, 2])))
    print(f"  geometry: layer '{tbl}', CRS units={units}, {how}")
    print(f"  sanity: median taluk size ~{side:.0f} km (Karnataka taluks are roughly 20-50 km across)")
    if not (3 < side < 200):
        print("  !! taluk size looks wrong -> CRS/unit problem. DO NOT trust --buffer-km until this is fixed.")
    return D, cent


# ----------------------------------------------------------------------------- DATA / FEATURES
def channel_group(name):
    n = name.lower()
    if any(k in n for k in ("vv", "vh", "sar", "radar")): return "radar"
    if any(k in n for k in ("elev", "dem", "slope", "aspect")): return "topo"
    if "minus" in n and "rain" in n: return "waterbalance"
    if any(k in n for k in ("rain", "chirps", "precip")): return "rain"
    if n in ("et", "et_mm") or n.startswith("et_") or "evapo" in n: return "et"
    if any(k in n for k in ("lst", "temp", "therm")): return "thermal"
    return "optical"


def load_taluk_table(index_path, extra_neg):
    seq_cols = [f"seq_y{y}_b{b}" for y in range(N_YEARS) for b in range(N_BANDS)]
    df = pd.read_parquet(index_path, columns=["taluk_id", "taluk_label"] + seq_cols)
    df = df[df["taluk_label"].isin([0, 1] + list(extra_neg))].reset_index(drop=True)
    df["taluk_id"] = df["taluk_id"].astype(str)
    assert df.groupby("taluk_id")["taluk_label"].nunique().max() == 1, "a taluk has two different labels"
    X = df[seq_cols].to_numpy(dtype=np.float32, copy=True).reshape(-1, N_YEARS, N_BANDS)
    X[~np.isfinite(X)] = np.nan                            # inf -> NaN
    taluks = sorted(df["taluk_id"].unique())
    cat = df.groupby("taluk_id")["taluk_label"].first().loc[taluks].to_numpy().astype(int)
    lab = (cat == 1).astype(int)
    rows = df.groupby("taluk_id").indices
    yearly = np.stack([np.nanmean(X[rows[t]], axis=0) for t in taluks])        # (taluks, years, bands)
    within_patch = np.nanmedian(np.nanstd(X, axis=1), axis=0)
    overall = np.nanstd(X.reshape(-1, N_BANDS), axis=0) + 1e-12
    static = within_patch / overall < 1e-3
    return np.array(taluks), lab, cat, yearly, static


def trend(y):
    m = np.isfinite(y)
    return np.polyfit(np.arange(len(y))[m], y[m], 1)[0] if m.sum() >= 3 else np.nan


def band_report(yearly, names, groups, static):
    """Check names -> band mapping: year-invariant (static) bands must be the topography bands."""
    mean = np.nanmean(yearly, axis=1)
    print("\n=== BAND CHECK (verify channel_names.json really matches seq_b0..b12) ===")
    print(f"{'b':>2} {'name':<14}{'group':<13}{'static':<7}{'min':>10}{'median':>10}{'max':>10}")
    bad = []
    for b in range(N_BANDS):
        col = mean[:, b]
        print(f"{b:>2} {names[b]:<14}{groups[b]:<13}{str(bool(static[b])):<7}{np.nanmin(col):>10.3g}"
              f"{np.nanmedian(col):>10.3g}{np.nanmax(col):>10.3g}")
        if static[b] != (groups[b] == "topo"):
            bad.append(names[b])
    if bad:
        print(f"  !! WARNING: static/non-static does not match the topo group for {bad}. The name->band mapping is probably\n"
              f"     wrong, so every group ablation below is mislabelled. Fix channel_names.json or pass --names.")


def build_features(yearly, groups, static):
    cols, names_, grp = [], [], []
    mean = np.nanmean(yearly, axis=1)
    for b in range(N_BANDS):
        cols.append(mean[:, b]); names_.append(f"b{b}_mean"); grp.append((b, "mean"))
        if groups[b] != "topo" and not static[b]:
            cols.append(np.array([trend(yearly[i, :, b]) for i in range(len(yearly))])); names_.append(f"b{b}_trend"); grp.append((b, "trend"))
    return np.column_stack(cols), grp


# ----------------------------------------------------------------------------- MODELS / LOTO
def lr_model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=SEED))


def loto_scores(XF, cent, lab, train_sets):
    n = len(lab)
    out = {k: np.full(n, np.nan) for k in list(XF) + ["kNN_coords"]}
    for i, tr in enumerate(train_sets):
        if min((lab[tr] == 0).sum(), (lab[tr] == 1).sum()) < MIN_TRAIN_PER_CLASS:
            continue
        for k, X in XF.items():
            out[k][i] = lr_model().fit(X[tr], lab[tr]).predict_proba(X[[i]])[0, 1]
        knn = KNeighborsClassifier(min(5, len(tr)), weights="distance").fit(cent[tr], lab[tr])
        out["kNN_coords"][i] = knn.predict_proba(cent[[i]])[0, 1]
    return out


def strat_boot(y, scores, n=2000):
    rng = np.random.default_rng(SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    out = {k: np.empty(n) for k in scores}
    for b in range(n):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        for k, s in scores.items():
            out[k][b] = roc_auc_score(y[i], s[i])
    return out


def ci(a):
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def misordered(y, s):
    pos, neg = s[y == 1], s[y == 0]
    return np.array([np.mean(pos < v) + 0.5 * np.mean(pos == v) for v in neg])


def perm_null(XF, cent, lab, train_sets, n_perm):
    null = {}
    for p in range(n_perm):
        lp = np.random.default_rng(SEED + 1000 + p).permutation(lab)
        for k, v in loto_scores(XF, cent, lp, train_sets).items():
            m = np.isfinite(v)
            if len(np.unique(lp[m])) == 2:
                null.setdefault(k, []).append(roc_auc_score(lp[m], v[m]))
    return {k: np.array(v) for k, v in null.items()}


# ----------------------------------------------------------------------------- REPORTING
def evaluate_block(title, sel, lab, cat, scores, null, summary):
    y = lab[sel]
    S = {k: v[sel] for k, v in scores.items()}
    n_neg, n_pos = int((y == 0).sum()), int((y == 1).sum())
    print("\n" + "=" * 92)
    print(f"  {title}: {len(y)} taluks ({n_neg} negative, {n_pos} OE; {n_neg * n_pos} pairs) | always-OE accuracy = {n_pos / len(y):.3f}")
    print("=" * 92)
    if n_neg < 2 or n_pos < 2:
        print("  too few taluks in one class.")
        return
    boots = strat_boot(y, S)
    print(f"  {'model':<16}{'AUC':>6}  {'95% CI':<15}{'perm-p':>7}{'Acc@.5':>8}{'BalAcc':>8}   TN FP FN TP")
    summary[title] = {}
    for k, s in S.items():
        a_ = roc_auc_score(y, s)
        lo, hi = ci(boots[k])
        pred = (s >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        pv = (1 + np.sum(null[k] >= a_)) / (1 + len(null[k])) if (null and k in null and len(null[k])) else None
        print(f"  {k:<16}{a_:>6.3f}  [{lo:.3f},{hi:.3f}]  {('%.3f' % pv) if pv is not None else '  n/a':>7}"
              f"{accuracy_score(y, pred):>8.3f}{balanced_accuracy_score(y, pred):>8.3f}   {tn:>2} {fp:>2} {fn:>2} {tp:>2}")
        summary[title][k] = dict(auc=a_, ci=[lo, hi], perm_p=pv, acc=float(accuracy_score(y, pred)),
                                 bal_acc=float(balanced_accuracy_score(y, pred)))
    print("  PAIRED bootstrap differences (same resamples):")
    for k in S:
        for ref in ("kNN_coords", "ALL"):
            if k == ref or ref not in S or (ref == "ALL" and k == "kNN_coords"):
                continue
            d = boots[k] - boots[ref]
            lo, hi = ci(d)
            verdict = "better" if lo > 0 else ("worse" if hi < 0 else "NOT distinguishable")
            print(f"    {k:<16} - {ref:<11} {d.mean():+.3f}  [{lo:+.3f},{hi:+.3f}]  {verdict}")
    print("  kNN_coords = geography only. A model not clearly above it adds nothing beyond location.")
    print("  perm-p: free label shuffle; labels are spatially clustered so this only means 'better than chance'.")
    if "ALL" in S:
        mis = misordered(y, S["ALL"])
        nc = cat[sel][y == 0]
        for cv in np.unique(nc):
            print(f"  ALL ranking error, {'strict Safe' if cv == 0 else 'extra negative (label %d)' % cv}: "
                  f"n={int((nc == cv).sum())}, mean={mis[nc == cv].mean():.2f} (0 perfect, 0.5 chance)")


def univariate_table(yearly, names, groups, lab, cent):
    mean = np.nanmean(yearly, axis=1)
    k = min(8, max(2, len(lab) // 5))
    coarse = KMeans(k, random_state=SEED, n_init=10).fit_predict(cent)
    rows = []
    for b in range(N_BANDS):
        f = mean[:, b]
        m = np.isfinite(f)
        if m.sum() < 5 or len(np.unique(lab[m])) < 2:
            continue
        a_ = roc_auc_score(lab[m], f[m])
        sc = f if a_ >= 0.5 else -f
        wa, ww = [], []
        for r in np.unique(coarse):
            mm = (coarse == r) & m
            if (lab[mm] == 0).sum() >= 1 and (lab[mm] == 1).sum() >= 1:
                wa.append(roc_auc_score(lab[mm], sc[mm])); ww.append(mm.sum())
        rho = max(abs(np.nan_to_num(spearmanr(f[m], cent[m, 0])[0])), abs(np.nan_to_num(spearmanr(f[m], cent[m, 1])[0])))
        rows.append(dict(band=names[b], group=groups[b], oriented_AUC=max(a_, 1 - a_), higher_in="OE" if a_ >= .5 else "Safe",
                         within_region_AUC=np.average(wa, weights=ww) if wa else np.nan, n_regions=len(wa), max_abs_rho_coords=rho))
    return pd.DataFrame(rows).sort_values("oriented_AUC", ascending=False)


# ----------------------------------------------------------------------------- MAIN
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="index.parquet")
    ap.add_argument("--gpkg", default="labeled_taluks.gpkg")
    ap.add_argument("--key", default="key", help="gpkg column that equals index.parquet taluk_id")
    ap.add_argument("--channels", default="channel_names.json")
    ap.add_argument("--names", default="", help="comma-separated 13 band names in seq_b0..b12 order (overrides the json)")
    ap.add_argument("--buffer-km", type=float, default=5.0)
    ap.add_argument("--also-negative", default="", help="label values to treat as NOT over-exploited, e.g. -1 for Semi-critical")
    ap.add_argument("--perm", type=int, default=20)
    ap.add_argument("--dl-scores", default="", help="loto_scores.csv written by clean_train_dl_v3.py (same buffer and labels!)")
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args(argv)
    extra = [int(v) for v in a.also_negative.split(",") if v.strip() != ""]

    print("=" * 92)
    print("  GROUNDWATER OE DETECTION - CLASSICAL BENCHMARK, buffered Leave-One-Taluk-Out (v4)")
    print("=" * 92)
    taluks, lab, cat, yearly, static = load_taluk_table(a.index, extra)
    n = len(taluks)
    print(f"  taluks={n} | strict Safe={(cat == 0).sum()} | extra negatives={np.isin(cat, extra).sum()} {extra or ''} | OE={(cat == 1).sum()}")

    if a.names:
        names = [s.strip() for s in a.names.split(",")]
    elif os.path.exists(a.channels):
        names = json.load(open(a.channels))
    else:
        names = [f"band_{i}" for i in range(N_BANDS)]
    names = list(names)[:N_BANDS]
    assert len(names) == N_BANDS, f"need {N_BANDS} band names, got {len(names)}"
    groups = [channel_group(x) for x in names]
    band_report(yearly, names, groups, static)

    D, cent = load_geometry(a.gpkg, list(taluks), a.key)
    F, fcols = build_features(yearly, groups, static)
    gi = lambda *g, kinds=("mean", "trend"): [j for j, (b, kd) in enumerate(fcols) if groups[b] in g and kd in kinds]
    sets = {"ALL": list(range(F.shape[1])), "ALL_means_only": gi(*set(groups), kinds=("mean",)),
            "ALL_minus_topo": [j for j, (b, kd) in enumerate(fcols) if groups[b] != "topo"],
            "optical+thermal": gi("optical", "thermal"), "radar_only": gi("radar"), "topo_only": gi("topo"),
            "rain+ET": gi("rain", "et", "waterbalance")}
    XF = {k: F[:, v] for k, v in sets.items() if len(v) > 0}
    print(f"  features: {F.shape[1]} columns (band means + trends of non-static bands) | sets: {list(XF)}")

    train_sets = [np.where((D[i] > a.buffer_km) & (np.arange(n) != i))[0] for i in range(n)]
    purged = np.array([n - 1 - len(t) for t in train_sets])
    print(f"  buffer={a.buffer_km} km: taluks purged per fold mean={purged.mean():.1f}, max={purged.max()} (of {n - 1})")

    scores = loto_scores(XF, cent, lab, train_sets)
    if a.dl_scores:
        d = pd.read_csv(a.dl_scores)
        d["taluk"] = d["taluk"].astype(str)
        scores["DL_GRU"] = np.array([dict(zip(d["taluk"], d["DL_GRU"])).get(t, np.nan) for t in taluks])
        print(f"  merged DL scores from {a.dl_scores} (make sure the same --buffer-km and --also-negative were used)")

    ok = np.all([np.isfinite(v) for v in scores.values()], axis=0)
    if (~ok).any():
        sk = [(t, int(l)) for t, l in zip(taluks[~ok], lab[~ok])]
        print(f"\n  Skipped {len(sk)} taluks (too few training negatives after purging, or no DL score): {sk}")
        if any(l == 0 for _, l in sk):
            print("  !! skipped NEGATIVE taluks make the results OPTIMISTIC (the hard cases were removed).")

    null = {}
    if a.perm > 0:
        print(f"\n  permutation null: {a.perm} label shuffles through the full LOTO pipeline ...")
        null = perm_null(XF, cent, lab, train_sets, a.perm)

    summary = {"buffer_km": a.buffer_km, "extra_negative_labels": extra, "n_taluks": int(n)}
    sel = np.where(ok)[0]
    evaluate_block("ALL negatives vs OE" if extra else "Safe vs OE", sel, lab, cat, scores, null, summary)
    if extra:
        evaluate_block("STRICT Safe vs OE (subset)", sel[np.isin(cat[sel], [0, 1])], lab, cat, scores, null, summary)

    print("\n=== UNIVARIATE BAND DIAGNOSTICS (taluk means) ===")
    ut = univariate_table(yearly, names, groups, lab, cent)
    print(ut.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("Note: bands are ranked and oriented using the labels, so with ~50 taluks the top AUCs are optimistic.\n"
          "High oriented_AUC with within_region_AUC ~0.5 and large max_abs_rho_coords = a regional/topographic proxy.")

    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame({"taluk": taluks, "category": cat, "label": lab, "n_purged": purged, **scores}).to_csv(os.path.join(a.out, "loto_scores.csv"), index=False)
    ut.to_csv(os.path.join(a.out, "band_diagnostics.csv"), index=False)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2, default=float)
    print(f"\n  Saved -> {a.out}/ (loto_scores.csv, band_diagnostics.csv, summary.json)")
    print("  Reminder: buffer / feature sets were fixed in advance; changing them after seeing results is tuning on the test set.")


if __name__ == "__main__":
    main()
