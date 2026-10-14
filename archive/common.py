import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json, sys, pickle, warnings
import numpy as np, pandas as pd, torch, zarr
from torch.utils.data import Dataset

ZARR_PATH, INDEX_PATH, CHANNEL_JSON = "hydrosentry.zarr", "index.parquet", "channel_names.json"
SEQ_PKL, NORM_NPZ, SEQ_NPZ = "cell_seq.pkl", "norm_stats.npz", "seq_stats.npz"
P = 32
SEQ_YEARS = list(range(2016, 2026))
BASE_CHANNELS = ["NDVI", "NDWI", "EVI", "LST", "rainfall_mm", "ET_mm", "VV", "VH",
                 "elevation", "slope", "aspect_sin", "aspect_cos"]
SEQ_CHANNELS = ["NDVI", "NDWI", "EVI", "LST", "rainfall_mm", "ET_mm", "VV", "VH"]

COLS = dict(taluk="taluk_id", year="year", row="row", col="col", label="taluk_label",
            gwl="gwl", stressed="stressed", well="well_id", fold="fold")

def load_index():
    df = pd.read_parquet(INDEX_PATH)
    df = df[[c for c in df.columns if not c.startswith("seq_")]]   # Phase-3 sequences are unused (and overflowed)
    df = df.drop(columns=[c for c in ("fold", "fold_spatial") if c in df.columns and c != COLS["fold"]])
    missing = [v for v in COLS.values() if v not in df.columns]
    if missing:
        raise KeyError(f"Columns not in index.parquet: {missing}. Available: {list(df.columns)}")
    df = df.rename(columns={v: k for k, v in COLS.items()})
    df["year"] = df["year"].astype(int)
    df["row"] = df["row"].astype(int)
    df["col"] = df["col"].astype(int)
    if os.path.exists(FOLDS_CSV):                                  # folds from make_folds.py override
        fm = pd.read_csv(FOLDS_CSV).set_index("taluk_id")["fold"]
        df["fold"] = df["taluk"].map(fm).fillna(-1).astype(int)
    w = df["well"]
    df["is_well"] = w.notna() & ~w.astype(str).isin(["", "None", "nan", "<NA>"])
    return df

def spec(use_sar=True):
    base = [c for c in BASE_CHANNELS if use_sar or c not in ("VV", "VH")]
    derived = ["ET_minus_rain"] + (["VV_minus_VH"] if use_sar else [])
    seq = [c for c in SEQ_CHANNELS if use_sar or c not in ("VV", "VH")]
    return base, derived, seq

class PatchStore:
    def __init__(self, use_sar=True):
        self.all_names = json.load(open(CHANNEL_JSON))
        self.base, self.derived, self.seq_names = spec(use_sar)
        self.idx = [self.all_names.index(n) for n in self.base]
        self.out_names = self.base + self.derived
        self.root = None

    def _root(self):
        if self.root is None:
            self.root = zarr.open_group(ZARR_PATH, mode="r")
        return self.root

    def raw_patch(self, taluk, year, row, col):
        arr = self._root()[f"{taluk}/{int(year)}"]
        _, H, W = arr.shape
        h = P // 2
        r0, c0 = int(row) - h, int(col) - h
        pr0, pc0, pr1, pc1 = max(r0, 0), max(c0, 0), min(r0 + P, H), min(c0 + P, W)
        out = np.full((len(self.idx), P, P), np.nan, np.float32)
        if pr1 > pr0 and pc1 > pc0:
            sub = np.asarray(arr[:, pr0:pr1, pc0:pc1])[self.idx].astype(np.float32)
            sub[~np.isfinite(sub)] = np.nan
            out[:, pr0 - r0:pr1 - r0, pc0 - c0:pc1 - c0] = sub
        return out

    def make_channels(self, x):
        d = {n: x[i].copy() for i, n in enumerate(self.base)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d["elevation"] = d["elevation"] - np.nanmean(d["elevation"])   # local relief only
        ch = [d[n] for n in self.base]
        ch.append(d["ET_mm"] - d["rainfall_mm"])
        if "VV" in d:
            ch.append(d["VV"] - d["VH"])
        return np.stack(ch).astype(np.float32)

class Normalizer:
    def __init__(self, out_names):
        d = np.load(NORM_NPZ)
        names = list(d["names"])
        ix = [names.index(n) for n in out_names]
        self.lo, self.hi, self.mean, self.std = [d[k][ix][:, None, None] for k in ("lo", "hi", "mean", "std")]

    def __call__(self, x):
        return (np.clip(x, self.lo, self.hi) - self.mean) / self.std

class SeqTable:
    def __init__(self, use_sar=True):
        s = np.load(SEQ_NPZ)
        self.d = None
        self.keep = [SEQ_CHANNELS.index(c) for c in spec(use_sar)[2]]
        self.mean, self.std = s["mean"][self.keep], s["std"][self.keep]

    def __getstate__(self):                     # do not pickle the big table into DataLoader workers
        st = self.__dict__.copy(); st["d"] = None
        return st

    def get(self, taluk, row, col):
        if self.d is None:
            self.d = pickle.load(open(SEQ_PKL, "rb"))
        a = self.d.get(f"{taluk}|{int(row)}|{int(col)}")
        if a is None:
            a = np.full((len(SEQ_YEARS), len(SEQ_CHANNELS)), np.nan, np.float32)
        a = (a[:, self.keep] - self.mean) / self.std
        return np.clip(np.nan_to_num(a), -5, 5).astype(np.float32)

def load_sample(store, norm, seqtab, taluk, year, row, col, augment, rng):
    x = norm(store.make_channels(store.raw_patch(taluk, year, row, col)))
    valid = np.isfinite(x).all(0, keepdims=True).astype(np.float32)
    x = np.concatenate([np.nan_to_num(x), valid], 0)
    if augment:
        x = np.rot90(x, int(rng.integers(4)), (1, 2))
        if rng.random() < 0.5:
            x = x[:, :, ::-1]
    seq = seqtab.get(taluk, row, col) if seqtab is not None else np.zeros((len(SEQ_YEARS), 1), np.float32)
    return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(seq)

class _Rng:
    _r = None
    def rng(self):
        if self._r is None:
            self._r = np.random.default_rng()   # OS entropy -> different in every worker
        return self._r

class GridDataset(Dataset, _Rng):
    def __init__(self, rows, store, norm, seqtab=None, augment=False):
        self.r = rows[["taluk", "year", "row", "col"]].values
        self.store, self.norm, self.seqtab, self.augment = store, norm, seqtab, augment
    def __len__(self): return len(self.r)
    def __getitem__(self, i):
        t, y, r, c = self.r[i]
        return load_sample(self.store, self.norm, self.seqtab, t, y, r, c, self.augment, self.rng())

class WellDataset(Dataset, _Rng):
    def __init__(self, rows, store, norm, seqtab, augment=False):
        self.df = rows.reset_index(drop=True)
        self.r = self.df[["taluk", "year", "row", "col"]].values
        self.y = self.df[["z", "stressed", "regmask", "clsmask"]].values.astype(np.float32)
        self.store, self.norm, self.seqtab, self.augment = store, norm, seqtab, augment
    def __len__(self): return len(self.r)
    def __getitem__(self, i):
        t, yr, r, c = self.r[i]
        x, s = load_sample(self.store, self.norm, self.seqtab, t, yr, r, c, self.augment, self.rng())
        return x, s, torch.from_numpy(self.y[i])

class BagDataset(Dataset, _Rng):
    def __init__(self, cells, labels, years, store, norm, seqtab, bag_size=64, min_valid=0.3):
        self.cells, self.years = cells, list(years)
        self.pos = [t for t, l in labels.items() if l == 1]
        self.neg = [t for t, l in labels.items() if l == 0]
        self.labels, self.bag_size, self.min_valid = labels, bag_size, min_valid
        self.store, self.norm, self.seqtab = store, norm, seqtab
    def __len__(self): return 10 ** 6
    def __getitem__(self, i):
        rng = self.rng()
        lst = self.pos if rng.random() < 0.5 else self.neg
        t = lst[int(rng.integers(len(lst)))]
        y = self.years[int(rng.integers(len(self.years)))]
        c = self.cells[t]
        sel = rng.choice(len(c), min(len(c), self.bag_size * 3 // 2), replace=False)
        X, S = [], []
        for j in sel:
            x, s = load_sample(self.store, self.norm, self.seqtab, t, y, c[j, 0], c[j, 1], True, rng)
            if float(x[-1].mean()) >= self.min_valid:
                X.append(x); S.append(s)
            if len(X) == self.bag_size:
                break
        if len(X) == 0:
            return self.__getitem__(i)
        while len(X) < self.bag_size:
            k = int(rng.integers(len(X))); X.append(X[k]); S.append(S[k])
        return torch.stack(X), torch.stack(S), torch.tensor(float(self.labels[t]))

GPKG, GPKG_KEY, FOLDS_CSV = "labeled_taluks.gpkg", "key", "folds.csv"
_G = None

def taluk_geoms():
    global _G
    if _G is None:
        import geopandas as gpd
        g = gpd.read_file(GPKG).to_crs(32643)
        _G = dict(zip(g[GPKG_KEY], g.geometry))
    return _G

def taluk_origin(root, taluk):
    a = root[taluk].attrs
    if "origin" in a:
        return a["origin"]
    b = taluk_geoms()[taluk].bounds
    return [np.floor(b[0] / 30) * 30, np.ceil(b[3] / 30) * 30]

def rc_to_xy(root, taluks, rows, cols):
    taluks, rows, cols = np.asarray(taluks), np.asarray(rows), np.asarray(cols)
    x, y = np.zeros(len(rows)), np.zeros(len(rows))
    for t in np.unique(taluks):
        m = taluks == t
        ox, oy = taluk_origin(root, t)
        x[m] = ox + (cols[m] + 0.5) * 30.0
        y[m] = oy - (rows[m] + 0.5) * 30.0
    return x, y

def build_norm(n=4000, seed=0):
    df = load_index()
    g = df[(~df.is_well) & (df.year >= 2016)].sample(n, random_state=seed)
    store = PatchStore(True)
    X = np.stack([store.make_channels(store.raw_patch(r.taluk, r.year, r.row, r.col)) for r in g.itertuples()])
    lo, hi, mean, std = [], [], [], []
    for c in range(X.shape[1]):
        v = X[:, c].ravel(); v = v[np.isfinite(v)]
        l, h = np.percentile(v, [1, 99]); v = np.clip(v, l, h)
        lo.append(l); hi.append(h); mean.append(v.mean()); std.append(v.std() + 1e-6)
    np.savez(NORM_NPZ, names=np.array(store.out_names), lo=np.array(lo, np.float32), hi=np.array(hi, np.float32),
             mean=np.array(mean, np.float32), std=np.array(std, np.float32))
    print("saved", NORM_NPZ, store.out_names)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "norm":
        build_norm()
