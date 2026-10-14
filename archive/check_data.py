import json, numpy as np, pandas as pd, zarr
from common import *

names = json.load(open(CHANNEL_JSON))
df = load_index(); root = zarr.open_group(ZARR_PATH, mode="r")

print("== 1. raw Zarr health (300 random grid patches) ==")
g = df[(~df.is_well) & (df.year >= 2016)].sample(300, random_state=0)
vals = [[] for _ in names]
for r in g.itertuples():
    a = root[f"{r.taluk}/{r.year}"]
    p = np.asarray(a[:, max(r.row - 16, 0):r.row + 16, max(r.col - 16, 0):r.col + 16]).astype(np.float32)
    for i in range(len(names)):
        vals[i].append(p[i].ravel())
rows = []
for i, n in enumerate(names):
    v = np.concatenate(vals[i]); f = v[np.isfinite(v)]
    rows.append(dict(channel=n, nan=np.isnan(v).mean(), inf=np.isinf(v).mean(),
                     zero=(f == 0).mean() if len(f) else np.nan,
                     min=f.min() if len(f) else np.nan, median=np.median(f) if len(f) else np.nan,
                     max=f.max() if len(f) else np.nan))
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n== 2. normalisation stats ==")
n = np.load(NORM_NPZ)
print(pd.DataFrame({k: n[k] for k in ("lo", "hi", "mean", "std")}, index=n["names"]).round(3))
s = np.load(SEQ_NPZ)
print("sequence stats finite:", bool(np.isfinite(s["mean"]).all() and np.isfinite(s["std"]).all()))

print("\n== 3. taluk keys vs gpkg ==")
print("taluks in index missing from gpkg:", sorted(set(df.taluk) - set(taluk_geoms())))

print("\n== 4. wells ==")
store = PatchStore(True)
w = df[df.is_well].reset_index(drop=True)
ok = [np.isfinite(store.make_channels(store.raw_patch(r.taluk, r.year, r.row, r.col))[:, P // 2, P // 2]).all()
      for r in w.itertuples()]
print(f"well centre pixel fully valid: {np.mean(ok):.1%} of {len(w)} (want ~100%)")
try:
    wp = pd.read_parquet("wells_premonsoon.parquet").drop_duplicates(["well_id", "year"])
    wp["well_id"] = wp.well_id.astype(str); w["well"] = w.well.astype(str)
    m = w.merge(wp[["well_id", "year", "x", "y"]], left_on=["well", "year"], right_on=["well_id", "year"])
    x, y = rc_to_xy(root, m.taluk, m.row, m.col)
    d = np.hypot(x - m.x.values, y - m.y.values)
    print(f"well coordinate -> pixel error (m): median {np.median(d):.1f}, 95th {np.percentile(d, 95):.1f}, "
          f"max {d.max():.1f}  (want all <= ~22)")
except Exception as e:
    print("coordinate check skipped:", e)
