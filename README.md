# 🌍 Groundwater Over-Exploitation Detection System (Pilot Study)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red)
![Status](https://img.shields.io/badge/Status-Pilot_Study-yellow)
![Validation](https://img.shields.io/badge/Validation-Buffered_LOTO-blue)

An empirical remote sensing and machine learning framework evaluating whether multi-sensor satellite time-series (Sentinel-1 SAR, Sentinel-2 Optical, Thermal LST, and Climate data: 2014–2023) add predictive information regarding Central Ground Water Board (CGWB) groundwater over-exploitation status beyond geographic location and topography across Karnataka, India.

---

## 📌 Abstract & Summary
We evaluated whether satellite-derived dynamics add information about CGWB groundwater over-exploitation status beyond location and terrain, using nested models (coordinates; + topography; + satellite dynamics) under buffered leave-one-taluk-out validation at 0-50 km. Complete imagery (2014-2023) was available for 44 over-exploited and 6 Safe taluks in Karnataka. At the pre-specified 25 km buffer, AUC was 0.485 for coordinates plus topography and 0.705 with satellite dynamics (ΔAUC = +0.220, 95% bootstrap CI -0.289 to +0.591). Intervals for the paired difference included zero at every buffer, and the satellite model's own intervals included chance, so the pilot is inconclusive. We also report a data audit showing that image missingness and patch count alone can separate the classes in an uncurated export, and the controls used to detect this.

---

## 🔍 Data Quality Audit & Artifact Control
A systematic preliminary audit revealed that in an uncurated Earth Engine export, image missingness (NaN fraction) alone achieved an AUC of 0.79 and patch count alone achieved an AUC of 0.77. 

Our audit traced the root cause: **most Safe taluks were missing from the Earth Engine export (94 of 119 exclusions were unexported or static-only, and 25 were cloud/border masked)**. Uncorrected, machine learning models exploit missingness patterns as an artificial classification shortcut rather than learning physical hydrological stress.

To prevent artifact learning, we instituted a data-quality gate:
1. **Patch Sampling:** Each taluk is sampled with 30 spatial patches from the taluk's multi-spectral grid index (which includes spatial grid points and observation well locations).
2. **Missingness Control:** Missing pixel fractions were audited across all taluks, confirming maximum missingness of 5.56% and an overall mean under 1.5%.
3. **Embedded Negative Controls:** Embedded control models (`CTRL_missingness` and `CTRL_patch_count`) were integrated directly into the cross-validation pipeline.

---

## 🛠️ Multi-Sensor Input Channels & Architecture
* **Satellite Observations (13 Channels, 2014–2023):**
  * **Optical Indices (Sentinel-2 MSI / Landsat 8):** NDVI, NDWI, EVI.
  * **Thermal & Climate Dynamics:** Land Surface Temperature (Landsat 8/9 TIRS & MODIS), CHIRPS seasonal precipitation sum, and MODIS MOD16A2GF Evapotranspiration.
  * **Microwave Radar (Sentinel-1 C-Band SAR):** VV, VH backscatter, and SAR validity mask.
  * **Topographical Constraints (SRTM GL1 30m DEM):** Elevation, slope, sine aspect, and cosine aspect.
* **Feature Engineering (26 Features):**
  * Features were mechanically computed as 10-year temporal means and linear trend slopes across all 13 cached channels (26 features total). For static terrain channels (elevation, slope, aspect sine, aspect cosine), temporal slopes were nominally zero, producing 8 topography columns (means and zero slopes).
  * Channel 8 contained a SAR acquisition validity mask, representing a potential missingness leak channel; however, the missingness control did not exceed chance (AUC 0.360–0.491).
  * Coordinates are bounding box centroids converted to approximate kilometres from the boundary geometry, and pairwise distances are computed as bounding-box Euclidean distances in kilometres.
* **Nested Feature Hierarchy:**
  * **$M0$ (Coordinates Only):** Centroid $X, Y$ coordinates.
  * **$M1$ (Coordinates + Topography):** Centroids + 8 terrain topography features.
  * **$M2$ (Coordinates + Topography + Satellite Dynamics):** Centroids + 8 terrain features + 26 satellite dynamic features.
* **Ground Truth Labels:** Central Ground Water Board (CGWB) statutory Dynamic Ground Water Resources Assessment (GWRA 2024/2025 assessment report). In this pilot cohort, 44 taluks are classified as Over-Exploited ($\text{SOE} > 100\%$) and 6 taluks are classified as Safe ($\text{SOE} < 70\%$).

---

## 📊 Pilot Benchmark Results (Frozen Nested Sweep)

Evaluated across Karnataka under geographically buffered Leave-One-Taluk-Out (LOTO) cross-validation ($N = 50$: 44 Over-Exploited, 6 Safe):

| Buffer (km) | $N$ (OE / Safe) | $M0$ Coords (AUC) | $M1$ Coords+Topo (AUC) | $M2$ +Satellite (AUC) | $\Delta\text{AUC}$ ($M2 - M1$) | 95% CI ($\Delta\text{AUC}$) | `CTRL_missingness` (NaN AUC) | Scientific Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **0 km** | 50 (44 / 6) | 0.621 | 0.557 | 0.754 | +0.197 | [-0.236, +0.504] | 0.360 | Inconclusive (CI spans zero) |
| **15 km** | 50 (44 / 6) | 0.538 | 0.424 | 0.659 | +0.235 | [-0.245, +0.568] | 0.417 | Inconclusive (CI spans zero) |
| **25 km** | 50 (44 / 6) | 0.591 | 0.485 | 0.705 | +0.220 | [-0.289, +0.591] | 0.426 | Inconclusive (CI spans zero) |
| **35 km** | 50 (44 / 6) | 0.598 | 0.504 | 0.716 | +0.212 | [-0.310, +0.591] | 0.443 | Inconclusive (CI spans zero) |
| **50 km** | 50 (44 / 6) | 0.587 | 0.530 | 0.739 | +0.208 | [-0.292, +0.582] | 0.491 | Inconclusive (CI spans zero) |

*Note: ΔAUC is computed on identical bootstrap resamples of taluks, applied to the pooled out-of-fold predictions. CTRL (NaN) measures whether missing data fraction alone predicts the label (target ~0.50).*

### Key Findings & Honest Assessment:
1. **Paired Contrast Spans Zero:** While point estimates favored the satellite model across all buffer distances (+0.197 to +0.235 ΔAUC), the 95% bootstrap confidence intervals for the paired contrast span zero at all buffers.
2. **Baseline Performance & Chance Overlap:** Coordinates plus topography ($M1$) performs at or below chance (0.424–0.557), indicating that part of $\Delta\text{AUC}$ arises from $M1$ performing poorly. Furthermore, the satellite model's ($M2$) own confidence intervals include chance (e.g., at 25 km: [0.375, 0.943]).
3. **Control Model Verification:** The missingness control model (`CTRL_missingness`) did not detectably differ from chance (AUC 0.36–0.49). With only 6 Safe taluks, neither reading is statistically distinguishable from 0.50, and we do not interpret it further. The patch-count control was 0.500 by construction (exactly 30 patches per taluk).

---

## 🔍 Limitations & Future Work

### Limitations:
* **Sample Imbalance in Usable Imagery:** Only 6 Safe taluks had usable imagery in this raster export cohort. The 6 Safe taluks are simply the ones that happened to be exported, so they may not be representative of Safe taluks statewide.
* **Ground Truth Definition:** CGWB groundwater over-exploitation classifications reflect an administrative accounting ledger (extraction versus recharge balancing) rather than physical water table depth visible from space.
* **Canopy Masking Hypotheses:** Localized discrepancies in heavily vegetated agricultural zones (such as Ajjampura) represent working hypotheses regarding canopy buffering that require ground verification.
* **Sample Neighborhood Representation:** Patches include observation-well locations, so taluk features describe the neighbourhood of sampled points, not the whole taluk.

### Future Work:
1. **Extend Earth Engine Export:** Scale imagery extraction to all 147 administrative taluks in Karnataka to obtain a balanced statewide sample ($\ge 50$ Safe taluks).
2. **Longitudinal Transition Modeling:** Test the model's ability to predict temporal transitions across consecutive CGWB assessment cycles (predicting taluks shifting from Safe to Over-Exploited over time).
3. **In-Situ Well Validation:** Validate patch-level inferences directly against continuous CGWB piezometric borewell telemetry using spatially blocked cross-validation by well.

---

## 🚀 Execution Guide

Run the frozen reviewer-grade nested benchmark across all buffer distances:
```powershell
foreach ($b in 0,15,25,35,50) {
  python stage1_nested_v5.py --buffer-km $b --force --out runs/pilot_$b
}
python make_pilot_table.py
```
*Outputs the pilot table to `runs/pilot_table.md`.*

---

## 🖥️ Launch Desktop GIS Dashboard

```powershell
python desktop_app.py
```

### Dashboard Workflow:
1. **Select Region:** Choose a target taluk from the dropdown menu.
2. **Scan Aquifer:** Click `🔍 Detect Over-Exploited Areas` to render high-resolution NDVI rasters overlaid with predicted stress heatmaps and view ranked GPS coordinates.
3. **Export Web Map:** Click `🌍 Export Interactive Web Map` to generate and launch an interactive Folium satellite map in your web browser.

---

## 🛠️ Technology Stack
* **Machine Learning:** Scikit-Learn (Balanced Logistic Regression, Spatial Cross-Validation)
* **Geospatial & Remote Sensing:** Zarr, PyProj, GeoPandas, Shapely, Rasterio
* **GUI & Visualization:** Tkinter, Matplotlib, Folium
* **Data Engineering:** NumPy, Pandas, PyArrow
