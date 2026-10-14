import zarr, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ZARR_PATH = "hydrosentry.zarr"
WELLS_PATH = "wells_premonsoon.parquet"
OUT_PATH = "index.parquet"
P = 32

print("Building index and temporal sequences...")
root = zarr.open_group(ZARR_PATH, mode="r")
years = list(range(2016, 2026))

# 1. Grid Index
grid_rows = []
for key in root.group_keys():
    g = root[key]
    H, W = g.attrs["shape"]
    mask = g["mask"][:]
    for r in range(16, H-16, P):
        for c in range(16, W-16, P):
            if mask[r-16:r+16, c-16:c+16].mean() >= 0.5:
                for y in years:
                    if str(y) in g:
                        grid_rows.append({"taluk_id": key, "year": y, "row": r, "col": c, 
                                          "taluk_label": g.attrs["label"], "split_type": "grid"})
idx = pd.DataFrame(grid_rows)
print(f"Grid index done: {len(idx)} rows.")

# 2. Well Index & Temporal Sequences
wells = pd.read_parquet(WELLS_PATH)
well_rows = []
temporal_feats = [] # For Phase 3 sequence building

for w in wells.itertuples():
    if w.taluk_id not in root: continue
    g = root[w.taluk_id]
    ox, oy = g.attrs["origin"]
    c, r = int((w.x-ox)/30), int((oy-w.y)/30)
    
    if g["mask"][r, c] and str(w.year) in g:
        well_rows.append({"taluk_id": w.taluk_id, "year": w.year, "row": r, "col": c, 
                          "taluk_label": g.attrs["label"], "gwl": w.gwl, "stressed": w.stressed, 
                          "well_id": w.well_id, "split_type": "well"})
        
        # Build 10-year sequence (Phase 3 requirement)
        seq = []
        for y_seq in range(2016, 2026):
            if str(y_seq) in g:
                patch = g[str(y_seq)][:, r-16:r+16, c-16:c+16]
                seq.append(np.nanmean(patch, axis=(1,2))) # Mean pooling for GRU
            else:
                seq.append(np.full(13, np.nan))
        temporal_feats.append(np.array(seq).flatten())

wdf = pd.DataFrame(well_rows)
if len(wdf) > 0:
    seq_df = pd.DataFrame(temporal_feats, columns=[f"seq_y{y}_b{b}" for y in range(10) for b in range(13)])
    wdf = pd.concat([wdf, seq_df], axis=1)
print(f"Well index done: {len(wdf)} rows.")

# 3. Define Folds (Grouped by Taluk) - Phase 3/5
# We only split on OE (1) and Safe (0). Semi-critical (-1) is pushed to a held-out test fold (fold -1).
df_all = pd.concat([idx, wdf], ignore_index=True)
df_all["fold"] = -1 

# Get unique taluks and their labels
taluk_labels = df_all.drop_duplicates("taluk_id")[["taluk_id", "taluk_label"]]
valid_taluks = taluk_labels[taluk_labels["taluk_label"].isin([0, 1])]

# Explicitly use scikit-learn for GroupKFold. If not installed, just use random taluk assignment as fallback.
try:
    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(valid_taluks, valid_taluks["taluk_label"], valid_taluks["taluk_id"])):
        val_taluk_ids = valid_taluks.iloc[val_idx]["taluk_id"].values
        df_all.loc[df_all["taluk_id"].isin(val_taluk_ids), "fold"] = fold
except ImportError:
    print("scikit-learn not installed. Using simple fallback grouping...")
    uq_taluks = valid_taluks["taluk_id"].values
    np.random.shuffle(uq_taluks)
    folds = np.array_split(uq_taluks, 5)
    for fold, f_taluks in enumerate(folds):
        df_all.loc[df_all["taluk_id"].isin(f_taluks), "fold"] = fold

df_all.to_parquet(OUT_PATH, index=False)
print(f"Index saved to {OUT_PATH} with {len(df_all)} rows, temporal sequences, and K-Fold CV splits.")
print("Taluks per fold:")
print(df_all.drop_duplicates('taluk_id').groupby('fold')['taluk_label'].value_counts())
