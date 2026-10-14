import json, pickle, warnings
import numpy as np, zarr
from common import *

if __name__ == "__main__":
    df = load_index()
    root = zarr.open_group(ZARR_PATH, mode="r")
    names = json.load(open(CHANNEL_JSON))
    sidx = [names.index(c) for c in SEQ_CHANNELS]
    h, out = P // 2, {}
    for taluk, g in df.groupby("taluk"):
        cells = [tuple(x) for x in g[["row", "col"]].drop_duplicates().values]
        S = {rc: np.full((len(SEQ_YEARS), len(SEQ_CHANNELS)), np.nan, np.float32) for rc in cells}
        grp = root[taluk]
        for yi, y in enumerate(SEQ_YEARS):
            if str(y) not in grp:
                continue
            arr = np.asarray(grp[str(y)][:])[sidx].astype(np.float32)
            arr[~np.isfinite(arr)] = np.nan
            for (r, c) in cells:
                patch = arr[:, max(r - h, 0):r + h, max(c - h, 0):c + h].reshape(len(sidx), -1)
                if patch.size == 0:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    S[(r, c)][yi] = np.nanmean(patch, axis=1)
        for (r, c), a in S.items():
            out[f"{taluk}|{r}|{c}"] = a
        print("seq done", taluk, len(cells))
    pickle.dump(out, open(SEQ_PKL, "wb"))
    allv = np.concatenate(list(out.values()), 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.savez(SEQ_NPZ, mean=np.nanmean(allv, 0), std=np.nanstd(allv, 0) + 1e-6)
    print("saved", SEQ_PKL, SEQ_NPZ)
