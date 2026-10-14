import pandas as pd
from common import *

df = load_index(); w = df[df.is_well]
tal = df.groupby("taluk").agg(label=("label", "first"), fold=("fold", "first"))
t = (tal.join(w.groupby("taluk").size().rename("well_rows"))
        .join(w.groupby("taluk").well.nunique().rename("wells"))
        .join(w.groupby("taluk").stressed.mean().rename("stressed_frac")))
t[["well_rows", "wells"]] = t[["well_rows", "wells"]].fillna(0).astype(int)
print(t.sort_values("well_rows").to_string())
t["usable_reg"] = t.well_rows >= 5
print("\n--- by taluk label (1=OE, 0=safe, -1=semi) ---")
print(t.groupby("label").agg(taluks=("well_rows", "size"), usable_taluks=("usable_reg", "sum"),
                             well_rows=("well_rows", "sum"), stressed_frac=("stressed_frac", "mean")))
print("\nwell rows per year:\n", w.groupby("year").size())
print("taluks with zero wells:", int((t.well_rows == 0).sum()))
