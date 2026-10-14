import numpy as np, pandas as pd, geopandas as gpd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

KEY, LAB, TALUK_COL = "key", "label_binary", "taluk_id"   
g = gpd.read_file("labeled_taluks.gpkg").to_crs(32643)
lab = g[g[LAB].isin([0, 1])].reset_index(drop=True)

b = gpd.GeoDataFrame(geometry=lab.geometry.buffer(1000))      # touching or within ~2 km
adj = gpd.sjoin(b, b, predicate="intersects")
n = len(lab)
A = csr_matrix((np.ones(len(adj)), (adj.index.values, adj.index_right.values)), shape=(n, n))
nc, comp = connected_components(A, directed=False)
sizes = np.bincount(comp)
print("neighbour-groups:", nc, "| sizes:", sorted(sizes, reverse=True))
print("largest group = %.0f%% of labelled taluks" % (100 * sizes.max() / n))

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
fold = np.full(n, -9)
for k, (_, te) in enumerate(sgkf.split(lab, lab[LAB], groups=comp)):
    fold[te] = k
lab["fold_spatial"] = fold
print(lab.groupby("fold_spatial")[LAB].agg(["count", "sum"]).rename(columns={"sum": "n_OE"}))

m = dict(zip(lab[KEY], lab["fold_spatial"]))
for k_ in g.loc[~g[LAB].isin([0, 1]), KEY]:
    m[k_] = -1                                                # semi-critical stays held out
idx = pd.read_parquet("index.parquet")
idx["fold_spatial"] = idx[TALUK_COL].map(m)
assert idx["fold_spatial"].notna().all()
idx.to_parquet("index.parquet")
print("Saved. Updated index.parquet with fold_spatial!")
