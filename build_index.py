"""
Phase 1 — Step 3: Build the Parquet index table
Produces: index.parquet
Depends on: hydrosentry.zarr (from build_zarr.py)
            wells_premonsoon.parquet (from wells_premonsoon.py)

This creates two kinds of samples:
  1. GRID samples: non-overlapping 32×32 patches (~0.96 km) tiling every taluk,
     used for SSL pretraining, MIL bags, and inference.
  2. WELL samples: patches centred on each well location,
     used for supervised GWL regression and stressed classification.

Both types are read by the same PatchDataset class.
"""
import zarr
import numpy as np
import pandas as pd
import json
import time

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
ZARR_PATH = "hydrosentry.zarr"
WELLS_PATH = "wells_premonsoon.parquet"
OUT_PATH = "index.parquet"

P = 32  # patch size in pixels (32 × 30m = 960m ≈ 1 km)

# Years for supervised training (Sentinel-2 + Sentinel-1 era)
SUPERVISED_YEARS = list(range(2016, 2026))

# All years (for SSL pretraining and trend features)
ALL_YEARS = [2005,2006,2007,2008,2009,2010,2011,
             2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025]


def main():
    print("=" * 60)
    print("BUILD INDEX TABLE")
    print("=" * 60)

    root = zarr.open_group(ZARR_PATH, mode="r")
    taluk_keys = list(root.group_keys())
    print(f"Taluks in Zarr store: {len(taluk_keys)}")

    half = P // 2  # 16 pixels

    # ── 1. Build grid index ──
    print("\nBuilding grid index (non-overlapping patches)...")
    t0 = time.time()
    grid_rows = []

    for key in taluk_keys:
        g = root[key]
        H, W = g.attrs["shape"]
        label = g.attrs["label"]
        mask = g["mask"][:]

        # Available years for this taluk
        available_years = [y for y in ALL_YEARS if str(y) in g]

        # Step through non-overlapping cells
        for r in range(half, H - half, P):
            for c in range(half, W - half, P):
                # Check that at least 50% of the patch is inside the taluk
                patch_mask = mask[r - half : r + half, c - half : c + half]
                if patch_mask.mean() < 0.5:
                    continue

                for yr in available_years:
                    grid_rows.append({
                        "taluk_id":    key,
                        "year":        yr,
                        "row":         r,
                        "col":         c,
                        "taluk_label": label,
                        "gwl":         np.nan,
                        "stressed":    np.nan,
                        "well_id":     None,
                        "split_type":  "grid",
                    })

    grid_df = pd.DataFrame(grid_rows)
    print(f"  Grid samples: {len(grid_df):,}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # ── 2. Build well index ──
    print("\nBuilding well index (patches centred on wells)...")
    t0 = time.time()

    wells = pd.read_parquet(WELLS_PATH)
    print(f"  Well-year records from parquet: {len(wells)}")

    well_rows = []
    skipped = 0

    for w in wells.itertuples():
        taluk_id = w.taluk_id
        if taluk_id not in root:
            skipped += 1
            continue

        g = root[taluk_id]
        ox, oy = g.attrs["origin"]  # (west, north) in UTM
        H, W = g.attrs["shape"]
        label = g.attrs["label"]

        # Convert UTM coordinates to pixel row/col
        # origin is top-left: ox = west edge, oy = north edge
        c = int((w.x - ox) / 30)   # column (east offset)
        r = int((oy - w.y) / 30)   # row (south offset from top)

        # Bounds check: patch must fit within the raster
        if r < half or r >= H - half or c < half or c >= W - half:
            skipped += 1
            continue

        # Check the well lands on a valid (inside-taluk) pixel
        if not g["mask"][r, c]:
            skipped += 1
            continue

        # Check the year exists in the Zarr store
        yr = int(w.year)
        if str(yr) not in g:
            skipped += 1
            continue

        well_rows.append({
            "taluk_id":    taluk_id,
            "year":        yr,
            "row":         r,
            "col":         c,
            "taluk_label": label,
            "gwl":         float(w.gwl),
            "stressed":    float(w.stressed),
            "well_id":     w.well_id,
            "split_type":  "well",
        })

    well_df = pd.DataFrame(well_rows)
    print(f"  Well samples matched: {len(well_df):,}")
    print(f"  Well samples skipped (outside bounds/mask/year): {skipped}")

    # ── 3. Combine and save ──
    index = pd.concat([grid_df, well_df], ignore_index=True)
    index.to_parquet(OUT_PATH, index=False)

    print(f"\n{'=' * 60}")
    print(f"SAVED: {OUT_PATH}")
    print(f"{'=' * 60}")

    # ── 4. Summary statistics ──
    print(f"\nTotal samples: {len(index):,}")
    print(f"  Grid: {len(grid_df):,}")
    print(f"  Well: {len(well_df):,}")

    print(f"\nGrid samples by taluk_label:")
    print(grid_df["taluk_label"].value_counts().to_string())

    print(f"\nWell samples by taluk_label:")
    if len(well_df) > 0:
        print(well_df["taluk_label"].value_counts().to_string())
        print(f"\nWell GWL stats:")
        print(well_df["gwl"].describe().to_string())
        print(f"\nWell stressed distribution:")
        print(well_df["stressed"].value_counts().to_string())
    else:
        print("  No well samples matched!")

    print(f"\nSamples per year (grid):")
    print(grid_df.groupby("year").size().to_string())

    print(f"\nSamples per year (wells):")
    if len(well_df) > 0:
        print(well_df.groupby("year").size().to_string())

    # ── 5. MIL bag summary ──
    print(f"\n── MIL Bag Summary (taluk-year combinations) ──")
    # Only for supervised years and labeled taluks (label >= 0)
    sup_grid = grid_df[(grid_df["year"].isin(SUPERVISED_YEARS)) &
                       (grid_df["taluk_label"] >= 0)]
    bags = sup_grid.groupby(["taluk_id", "year", "taluk_label"]).size().reset_index(name="bag_size")
    print(f"  Total bags (labeled, 2016-2025): {len(bags)}")
    print(f"  Bag size stats:\n{bags['bag_size'].describe().to_string()}")
    print(f"  Bags by label:")
    print(f"    OE (1):   {len(bags[bags['taluk_label']==1])}")
    print(f"    Safe (0): {len(bags[bags['taluk_label']==0])}")


if __name__ == "__main__":
    main()
