import warnings; warnings.filterwarnings("ignore")
import os, json, numpy as np, pandas as pd, geopandas as gpd, torch, zarr, shapely
from shapely.ops import unary_union
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from pyproj import Transformer
from common import *
from model import Encoder

MANUAL = "manual_dataset&ground_truth/gwl_manual_quarterly_cgwb_ka_2021_2025.csv"
FEAT_YEARS = [2022, 2023, 2024]
MIN_READINGS, MIN_WELLS, ALPHA, BUFFER_M, N_PERM = 6, 4, 1000.0, 3000.0, 300
OUT = "runs/final"; os.makedirs(OUT, exist_ok=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
store = PatchStore(True); norm = Normalizer(store.out_names); C = len(store.out_names)
enc = Encoder(C + 1, 32).to(dev)
enc.load_state_dict(torch.load("pretrained_sar.pt", map_location=dev)); enc.eval()
rng = np.random.default_rng(0)
root = zarr.open_group(ZARR_PATH, mode="r")

@torch.no_grad()
def embed_rows(rows, years):
    n, D = len(rows), enc.out_dim
    E, Fm, V = np.zeros((n, D)), np.zeros((n, 2 * C)), np.zeros(n)
    for y in years:
        for i in range(0, n, 256):
            ch = rows[i:i + 256]
            xb = torch.stack([load_sample(store, norm, None, t, y, r, c, False, rng)[0] for t, r, c in ch])
            E[i:i + len(ch)] += enc(xb.to(dev))[1].cpu().numpy()
            f = xb[:, :C]
            Fm[i:i + len(ch)] += torch.cat([f.mean((2, 3)), f.std((2, 3))], 1).numpy()
            V[i:i + len(ch)] += xb[:, -1].mean((1, 2)).numpy()
        print("embedded year", y, "rows", n)
    k = len(years)
    return E / k, Fm / k, V / k

# ---------------- 1. well targets from ALL quarters ----------------
m = pd.read_csv(MANUAL)
LV = [c for c in m.columns if c.startswith("Groundwater Level")][0]
m["t"] = pd.to_datetime(m["Data Acquisition Time"], dayfirst=True, errors="coerce")
m[LV] = pd.to_numeric(m[LV], errors="coerce")
m = m.dropna(subset=["t", LV, "Latitude", "Longitude"])
m = m[(m[LV] > 0) & (m[LV] < 100)]
wl = (m.groupby("Station").agg(depth=(LV, "median"), n=(LV, "size"),
      lat=("Latitude", "first"), lon=("Longitude", "first")).reset_index())
wl = wl[wl.n >= MIN_READINGS]
print("wells with >=%d readings statewide:" % MIN_READINGS, len(wl))
assert len(wl) > 50, "check the level column / sign convention"

tg = gpd.read_file(GPKG).to_crs(32643)[[GPKG_KEY, "geometry"]]
pts = gpd.GeoDataFrame(wl, geometry=gpd.points_from_xy(wl.lon, wl.lat), crs=4326).to_crs(32643)
pts = gpd.sjoin(pts, tg, predicate="within").rename(columns={GPKG_KEY: "taluk"})
pts["x"], pts["y_utm"] = pts.geometry.x, pts.geometry.y
recs, masks = [], {}
for r in pts.itertuples():
    if r.taluk not in root: continue
    ox, oy = taluk_origin(root, r.taluk)
    c_, r_ = int((r.x - ox) // 30), int((oy - r.y_utm) // 30)
    mk = masks.setdefault(r.taluk, root[r.taluk]["mask"][:])
    if 0 <= r_ < mk.shape[0] and 0 <= c_ < mk.shape[1] and mk[r_, c_]:
        recs.append((r.Station, r.taluk, r_, c_, r.x, r.y_utm, r.depth, r.n))
W = pd.DataFrame(recs, columns=["well", "taluk", "row", "col", "x", "y_utm", "depth", "n"])

df = load_index()
tal = df.groupby("taluk").agg(fold=("fold", "first"), label=("label", "first"))
foldmap = tal.fold.to_dict()
W["fold"] = W.taluk.map(foldmap)
cnt = W.groupby("taluk").well.transform("count")
W["elig"] = cnt >= MIN_WELLS
W["ly"] = np.log1p(W.depth)
W["y"] = W.ly - W.groupby("taluk").ly.transform("mean")
print(f"wells inside the 54 taluks: {len(W)} | eligible (taluk has >= {MIN_WELLS} wells): {int(W.elig.sum())} "
      f"in {W[W.elig].taluk.nunique()} taluks")
print(W[W.elig].groupby("fold").size().rename("eligible wells per fold").to_string())

# ---------------- 2. embeddings (cached for cells) ----------------
grid = df[~df.is_well].drop_duplicates(["taluk", "row", "col"])[["taluk", "row", "col"]].reset_index(drop=True)
if os.path.exists("wellmap_cache.npz"):
    z = np.load("wellmap_cache.npz"); Ec, Fc, Vc = z["E"], z["F"], z["V"]
    assert len(Ec) == len(grid), "cache does not match grid; delete wellmap_cache.npz"
else:
    Ec, Fc, Vc = embed_rows(list(zip(grid.taluk, grid.row, grid.col)), FEAT_YEARS)
    np.savez("wellmap_cache.npz", E=Ec, F=Fc, V=Vc)
EW, FW, _ = embed_rows(list(zip(W.taluk, W.row, W.col)), FEAT_YEARS)

keep = Vc >= 0.3
cells = grid[keep].reset_index(drop=True); Ec, Fc = Ec[keep], Fc[keep]

def center(A_cells, A_wells):
    d = pd.DataFrame(A_cells); d["t"] = cells.taluk.values
    mu = d.groupby("t").mean()
    return A_cells - mu.loc[cells.taluk].values, A_wells - mu.loc[W.taluk].values

Ec_c, EW_c = center(Ec, EW)
Fc_c, FW_c = center(Fc, FW)
FEATS = {"embedding": (EW_c, Ec_c), "handcrafted": (FW_c, Fc_c)}

# ---------------- 3. leakage-controlled splits ----------------
G = taluk_geoms()
def train_mask(val_taluks, folds_ok):
    zone = unary_union([G[t] for t in val_taluks]).buffer(BUFFER_M)
    inside = shapely.contains_xy(zone, W.x.values, W.y_utm.values)
    return (W.elig & W.fold.isin(folds_ok) & ~W.taluk.isin(val_taluks) & ~inside).values

splits = {}
for k in range(5):
    vt = {t for t, f in foldmap.items() if f == k}
    splits[k] = (train_mask(vt, [f for f in range(5) if f != k]), (W.elig & (W.fold == k)).values)

def oof(X, ytr):
    p = np.full(len(W), np.nan)
    for k, (tr, te) in splits.items():
        mdl = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA)).fit(X[tr], ytr[tr])
        p[te] = mdl.predict(X[te])
    return p

def metrics(p):
    ev = ~np.isnan(p); w = W[ev].reset_index(drop=True); pp = p[ev]
    rho = spearmanr(pp, w.y)[0]
    per = [spearmanr(pp[ix], w.y.values[ix])[0] for ix in w.groupby("taluk").indices.values() if len(ix) >= 6]
    hi = (w.groupby("taluk").y.rank(pct=True).values > 0.75).astype(int)
    auc = roc_auc_score(hi, pp) if 0 < hi.sum() < len(hi) else np.nan
    return dict(n=int(ev.sum()), spearman=float(rho), mean_per_taluk_spearman=float(np.nanmean(per)) if per else np.nan,
                auc_top25=float(auc))

def perm_y():
    y = W.y.values.copy()
    for ix in W[W.elig].groupby("taluk").indices.values():
        pass
    for t, g in W[W.elig].groupby("taluk"):
        y[g.index.values] = rng.permutation(y[g.index.values])
    return y

res = {}
for name, (XW, XC) in FEATS.items():
    obs = metrics(oof(XW, W.y.values)); null = []
    for _ in range(N_PERM):
        pm = oof(XW, perm_y())
        ev = ~np.isnan(pm)
        null.append(spearmanr(pm[ev], W.y.values[ev])[0])
    obs["perm_p_spearman"] = float((1 + (np.array(null) >= obs["spearman"]).sum()) / (1 + N_PERM))
    obs["null_95th_pct"] = float(np.percentile(null, 95))
    res[name] = obs
    print(name, json.dumps(obs, indent=1))

# semi-critical taluks (never trained on)
sem_t = {t for t, f in foldmap.items() if f == -1}
tr_s = train_mask(sem_t, list(range(5)))
te_s = (W.elig & (W.fold == -1)).values
if te_s.sum() >= 10:
    mdl = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA)).fit(EW_c[tr_s], W.y.values[tr_s])
    res["semi_critical_spearman"] = float(spearmanr(mdl.predict(EW_c[te_s]), W.y.values[te_s])[0])
    res["semi_critical_n"] = int(te_s.sum())
json.dump(res, open(f"{OUT}/summary.json", "w"), indent=2)

# ---------------- 4. out-of-fold cell scores for the map ----------------
parts = []
def predict_cells(tr, val_taluks, role, k):
    mdl = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA)).fit(EW_c[tr], W.y.values[tr])
    sel = cells.taluk.isin(val_taluks).values
    c = cells[sel].copy(); c["score"] = mdl.predict(Ec_c[sel]); c["role"], c["model_fold"] = role, k
    parts.append(c)
for k in range(5):
    predict_cells(splits[k][0], {t for t, f in foldmap.items() if f == k}, "val", k)
predict_cells(tr_s, sem_t, "semi", -1)
out = pd.concat(parts, ignore_index=True)
out["pct"] = out.groupby("taluk").score.rank(pct=True) * 100
x, y = rc_to_xy(root, out.taluk.values, out.row.values, out.col.values)
out["lon"], out["lat"] = Transformer.from_crs(32643, 4326, always_xy=True).transform(x, y)
out.to_parquet(f"{OUT}/cells_wellhead.parquet")
W[["well", "taluk", "row", "col", "depth", "y", "elig", "fold"]].to_csv(f"{OUT}/wells_used.csv", index=False)
print("saved", len(out), "cells ->", f"{OUT}/cells_wellhead.parquet")
