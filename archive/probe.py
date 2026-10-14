import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from common import *
from model import Encoder
from train_cv import bootstrap_auc

dev = "cuda" if torch.cuda.is_available() else "cpu"
store = PatchStore(True); norm = Normalizer(store.out_names)
enc = Encoder(len(store.out_names) + 1, 32).to(dev)
enc.load_state_dict(torch.load("pretrained_sar.pt", map_location=dev)); enc.eval()
rng = np.random.default_rng(0)

@torch.no_grad()
def embed(rows):
    out = []
    for i in range(0, len(rows), 128):
        xs = [load_sample(store, norm, None, t, y, r, c, False, rng)[0] for t, y, r, c in rows[i:i + 128]]
        out.append(enc(torch.stack(xs).to(dev))[1].cpu().numpy())
    return np.concatenate(out)

df = load_index()
tal = df.groupby("taluk").agg(fold=("fold", "first"), label=("label", "first")).reset_index()

# ---- taluk embeddings: mean over 150 random 2025 cells (label-free) ----
grid = df[(~df.is_well) & (df.year == 2025)]
Tm = {}
for t in tal.taluk:
    g = grid[grid.taluk == t]; g = g.sample(min(150, len(g)), random_state=0)
    Tm[t] = embed(g[["taluk", "year", "row", "col"]].values.tolist()).mean(0)
print("taluk embeddings done")

# ---------------- TALUK LEVEL ----------------
lab = tal[tal.label.isin([0, 1])].reset_index(drop=True)
X = np.stack([Tm[t] for t in lab.taluk]); y0 = lab.label.values; fo = lab.fold.values

def oof_taluk(y):
    s = np.zeros(len(y))
    for k in np.unique(fo):
        tr, te = fo != k, fo == k
        if len(set(y[tr])) < 2: continue
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, class_weight="balanced", max_iter=2000))
        s[te] = m.fit(X[tr], y[tr]).decision_function(X[te])
    return s

auc = roc_auc_score(y0, oof_taluk(y0)); lo, hi = bootstrap_auc(y0, oof_taluk(y0))
null = []
for _ in range(300):
    yp = rng.permutation(y0)
    if len(set(yp)) == 2: null.append(roc_auc_score(yp, oof_taluk(yp)))
print(f"TALUK  AUC {auc:.3f}  95% CI [{lo:.2f}, {hi:.2f}]  permutation p = {(np.array(null) >= auc).mean():.3f}")

# ---------------- WELL LEVEL (within-taluk) ----------------
w = df[df.is_well & (df.fold >= 0)].copy()
g = w.groupby("taluk")["gwl"]; sd, n = g.transform("std"), g.transform("count")
w["z"] = (w.gwl - g.transform("mean")) / sd
w = w[(n >= 5) & (sd > 1e-6) & w.z.notna()].reset_index(drop=True)
w["z"] = w.z.clip(-3, 3)
EW = embed(w[["taluk", "year", "row", "col"]].values.tolist())
EWc = EW - np.stack([Tm[t] for t in w.taluk])            # remove the taluk-level signal
wf, z0 = w.fold.values, w.z.values

def oof_well(z):
    p = np.zeros(len(z))
    for k in np.unique(wf):
        tr, te = wf != k, wf == k
        m = make_pipeline(StandardScaler(), Ridge(alpha=1000)).fit(EWc[tr], z[tr])
        p[te] = m.predict(EWc[te])
    return p

def perm_within(z):
    z = z.copy()
    for _, ix in w.groupby("taluk").indices.items(): z[ix] = rng.permutation(z[ix])
    return z

rho = spearmanr(oof_well(z0), z0)[0]
null = [spearmanr(oof_well(perm_within(z0)), z0)[0] for _ in range(200)]
print(f"WELL   within-taluk Spearman {rho:.3f}  (n={len(w)})  permutation p = {(np.array(null) >= rho).mean():.3f}")
