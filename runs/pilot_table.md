# Pilot Benchmark Summary: Nested Buffer Sweep

> **Note:** All runs carry the pilot caveat: with only 6 Safe taluks with complete imagery,
> confidence intervals for the paired contrast (Delta AUC = M2 - M1) span zero.
> Treat results as an exploratory pilot until Earth Engine export adds more Safe taluks.

| Buffer (km) | N (OE / Safe) | M0 Coords (AUC) | M1 Coords+Topo (AUC) | M2 +Satellite (AUC) | Delta AUC (M2 - M1) | 95% CI (Delta AUC) | CTRL (NaN AUC) | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 0 km | 50 (44 / 6) | 0.621 | 0.557 | 0.754 | +0.197 | [-0.236, +0.504] | 0.360 | Inconclusive (CI spans zero) |
| 15 km | 50 (44 / 6) | 0.538 | 0.424 | 0.659 | +0.235 | [-0.245, +0.568] | 0.417 | Inconclusive (CI spans zero) |
| 25 km | 50 (44 / 6) | 0.591 | 0.485 | 0.705 | +0.220 | [-0.289, +0.591] | 0.426 | Inconclusive (CI spans zero) |
| 35 km | 50 (44 / 6) | 0.598 | 0.504 | 0.716 | +0.212 | [-0.310, +0.591] | 0.443 | Inconclusive (CI spans zero) |
| 50 km | 50 (44 / 6) | 0.587 | 0.530 | 0.739 | +0.208 | [-0.292, +0.582] | 0.491 | Inconclusive (CI spans zero) |

### Key Methodological Notes:
1. **Nested Models:** M0 = Coordinates; M1 = Coordinates + Topography; M2 = Coordinates + Topography + Satellite Dynamics.
2. **Paired Contrast:** ΔAUC is computed on identical bootstrap resamples of taluks, applied to the pooled out-of-fold predictions.
3. **Missingness Control:** CTRL (NaN) measures whether missing data fraction alone predicts the label (target ~0.50).