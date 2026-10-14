import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from common import COLS, INDEX_PATH, FOLDS_CSV

idx = pd.read_parquet(INDEX_PATH, columns=[COLS["taluk"], COLS["label"], COLS["well"]])
idx.columns = ["taluk_id", "label", "well"]
idx["is_well"] = idx.well.notna() & ~idx.well.astype(str).isin(["", "None", "nan", "<NA>"])
tal = idx.groupby("taluk_id").agg(label=("label", "first"), n_well_rows=("is_well", "sum")).reset_index()
lab = tal[tal.label.isin([0, 1])].reset_index(drop=True)

def assign(seed):
    f = np.full(len(lab), -9)
    sg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for k, (_, te) in enumerate(sg.split(lab, lab.label, groups=lab.taluk_id)):
        f[te] = k
    return f

best = None
for seed in range(300):
    f = assign(seed)
    if min(((f == k) & (lab.label == 0)).sum() for k in range(5)) < 1:
        continue                                    # every fold needs a safe taluk
    score = np.std([lab.n_well_rows[f == k].sum() for k in range(5)])
    if best is None or score < best[0]:
        best = (score, seed, f)
assert best is not None, "could not find a fold assignment with a safe taluk in every fold"
lab["fold"] = best[2]
print("seed", best[1])
print(lab.groupby("fold").agg(taluks=("taluk_id", "count"), n_OE=("label", "sum"),
      n_safe=("label", lambda s: (s == 0).sum()), well_rows=("n_well_rows", "sum"),
      safe_well_rows=("n_well_rows", lambda s: s[lab.loc[s.index, "label"] == 0].sum())))
semi = tal[~tal.label.isin([0, 1])].assign(fold=-1)
pd.concat([lab[["taluk_id", "fold"]], semi[["taluk_id", "fold"]]]).to_csv(FOLDS_CSV, index=False)
print("saved", FOLDS_CSV, "| semi-critical taluks held out:", len(semi))
