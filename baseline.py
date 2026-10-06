import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from common import *
from train_cv import bootstrap_auc

store = PatchStore(True); norm = Normalizer(store.out_names)
df = load_index(); nC = len(store.out_names)

def patch_feats(t, y, r, c):
    x = norm(store.make_channels(store.raw_patch(t, y, r, c)))
    f = x.reshape(x.shape[0], -1)
    return np.concatenate([np.nanmean(f, 1), np.nanstd(f, 1)])

# ---------- well level (within-taluk z target, same as the deep model) ----------
w = df[df.is_well & (df.fold >= 0)].reset_index(drop=True)
g = w.groupby("taluk")["gwl"]; sd, n = g.transform("std"), g.transform("count")
w["z"] = (w.gwl - g.transform("mean")) / sd
w["reg"] = (n >= 5) & (sd > 1e-6) & w.z.notna()
w["z"] = w.z.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-3, 3)
Xw = np.array([patch_feats(r.taluk, r.year, r.row, r.col) for r in w.itertuples()])
w["z_pred"], w["s_pred"] = np.nan, np.nan
for k in sorted(w.fold.unique()):
    tr, te = (w.fold != k).values, (w.fold == k).values
    m = tr & w.reg.values
    reg = HistGradientBoostingRegressor(max_depth=3, max_iter=150, learning_rate=0.05, random_state=0).fit(Xw[m], w.z[m])
    clf = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05, random_state=0).fit(Xw[tr], w.stressed[tr].astype(int))
    w.loc[te, "z_pred"] = reg.predict(Xw[te]); w.loc[te, "s_pred"] = clf.predict_proba(Xw[te])[:, 1]
mm = w.reg.values
print("WELL  spearman(z):", round(spearmanr(w.z_pred[mm], w.z[mm])[0], 3),
      "| stress AUC:", round(roc_auc_score(w.stressed, w.s_pred), 3),
      "| AP:", round(average_precision_score(w.stressed, w.s_pred), 3),
      "| AP baseline (prevalence):", round(w.stressed.mean(), 3))

# ---------- taluk level, by feature group (shortcut check) ----------
tal = df.groupby("taluk").agg(fold=("fold", "first"), label=("label", "first")).reset_index()
tal = tal[tal.label.isin([0, 1])].reset_index(drop=True)
grid = df[(~df.is_well) & (df.year == 2025)]
TF = []
for t in tal.taluk:
    gg = grid[grid.taluk == t]; gg = gg.sample(min(200, len(gg)), random_state=0)
    F = np.array([patch_feats(t, 2025, r.row, r.col) for r in gg.itertuples()])
    TF.append(np.nanmean(F[:, :nC], 0))
TF = np.array(TF); names = store.out_names
groups = {
    "all": names,
    "climate_only": ["rainfall_mm", "ET_mm", "ET_minus_rain", "LST"],
    "terrain_only": ["slope", "aspect_sin", "aspect_cos"],
    "veg_sar_only": ["NDVI", "NDWI", "EVI", "VV", "VH", "VV_minus_VH"],
    "no_climate": [c for c in names if c not in ("rainfall_mm", "ET_mm", "ET_minus_rain", "LST")],
}
y = tal.label.values
for gname, cols in groups.items():
    X = TF[:, [names.index(c) for c in cols]]; oof = np.zeros(len(tal))
    for k in sorted(tal.fold.unique()):
        tr, te = (tal.fold != k).values, (tal.fold == k).values
        mdl = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000))
        oof[te] = mdl.fit(X[tr], y[tr]).decision_function(X[te])
    lo, hi = bootstrap_auc(y, oof)
    print(f"TALUK {gname:13s} AUC {roc_auc_score(y, oof):.3f}  95% CI [{lo:.2f}, {hi:.2f}]")
