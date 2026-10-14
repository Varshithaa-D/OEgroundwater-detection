import numpy as np, matplotlib.pyplot as plt
from common import *

if __name__ == "__main__":
    df = load_index(); store = PatchStore(True)
    w = df[df.is_well]
    ok = []
    for r in w.itertuples():
        p = store.make_channels(store.raw_patch(r.taluk, r.year, r.row, r.col))
        ok.append(np.isfinite(p[:, P // 2, P // 2]).all())
    print(f"well centre pixel valid in {np.mean(ok):.1%} of {len(w)} well samples (should be ~100%)")

    fig, ax = plt.subplots(2, 4, figsize=(14, 7))
    for i, r in enumerate(w.sample(4, random_state=1).itertuples()):
        p = store.make_channels(store.raw_patch(r.taluk, r.year, r.row, r.col))
        ax[0, i].imshow(p[0], cmap="YlGn"); ax[0, i].plot(P // 2, P // 2, "r+")
        ax[0, i].set_title(f"{str(r.taluk)[:20]} {r.year}\nGWL={r.gwl:.1f} NaN={np.isnan(p[0]).mean():.2f}", fontsize=8)
        ax[1, i].imshow(p[store.out_names.index("elevation")], cmap="terrain")
    plt.tight_layout(); plt.savefig("sanity_check.png")
    print("Saved sanity_check.png")
