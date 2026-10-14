"""
make_pilot_table.py - Aggregates Nested Buffer Sweep Runs into a Single Pilot Table
====================================================================================
Reads runs/pilot_* directories and generates:
  - runs/pilot_table.md
  - Prints clean markdown table to console

Usage:
  python make_pilot_table.py
"""

import os, glob, json
import pandas as pd

def build_table():
    search_pattern = os.path.join("runs", "pilot_*", "metrics.json")
    files = glob.glob(search_pattern)

    if not files:
        print(f"[-] No metric files found matching '{search_pattern}'. Run the sweep first.")
        return

    records = []
    for f in files:
        try:
            with open(f, "r") as fp:
                d = json.load(fp)
                records.append(d)
        except Exception as e:
            print(f"[!] Warning reading {f}: {e}")

    if not records:
        print("[-] No records parsed.")
        return

    # Sort records by buffer distance
    records = sorted(records, key=lambda x: x["buffer_km"])

    table_md = []
    table_md.append("# Pilot Benchmark Summary: Nested Buffer Sweep")
    table_md.append("")
    table_md.append("> **Note:** All runs carry the pilot caveat: with only 6 Safe taluks with complete imagery,")
    table_md.append("> confidence intervals for the paired contrast (Delta AUC = M2 - M1) span zero.")
    table_md.append("> Treat results as an exploratory pilot until Earth Engine export adds more Safe taluks.")
    table_md.append("")
    table_md.append("| Buffer (km) | N (OE / Safe) | M0 Coords (AUC) | M1 Coords+Topo (AUC) | M2 +Satellite (AUC) | Delta AUC (M2 - M1) | 95% CI (Delta AUC) | CTRL (NaN AUC) | Status |")
    table_md.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|")

    for r in records:
        b = int(r['buffer_km']) if r['buffer_km'] == int(r['buffer_km']) else r['buffer_km']
        n_str = f"{r['n_total']} ({r['n_oe']} / {r['n_safe']})"
        m0_str = f"{r['auc_m0']:.3f}"
        m1_str = f"{r['auc_m1']:.3f}"
        m2_str = f"{r['auc_m2']:.3f}"
        delta_str = f"{r['delta_m2_m1']:+.3f}"
        ci_str = f"[{r['ci_delta'][0]:+.3f}, {r['ci_delta'][1]:+.3f}]"
        ctrl_str = f"{r['auc_ctrl_nan']:.3f}"
        status = r['status']
        table_md.append(f"| {b} km | {n_str} | {m0_str} | {m1_str} | {m2_str} | {delta_str} | {ci_str} | {ctrl_str} | {status} |")

    table_md.append("")
    table_md.append("### Key Methodological Notes:")
    table_md.append("1. **Nested Models:** M0 = Coordinates; M1 = Coordinates + Topography; M2 = Coordinates + Topography + Satellite Dynamics.")
    table_md.append("2. **Paired Contrast:** ΔAUC is computed on identical bootstrap resamples of taluks, applied to the pooled out-of-fold predictions.")
    table_md.append("3. **Missingness Control:** CTRL (NaN) measures whether missing data fraction alone predicts the label (target ~0.50).")

    full_text = "\n".join(table_md)

    os.makedirs("runs", exist_ok=True)
    out_path = os.path.join("runs", "pilot_table.md")
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write(full_text)

    print("\n" + "=" * 80)
    print("PILOT TABLE GENERATED:")
    print("=" * 80)
    for line in table_md[4:]:
        print(line)
    print("=" * 80)
    print(f"\n[Saved] Pilot table saved to: {out_path}\n")

if __name__ == "__main__":
    build_table()
