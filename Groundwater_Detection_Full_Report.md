# Detecting Groundwater Over-Exploitation from Satellite Remote Sensing: A Data Audit and Buffered Pilot Benchmark in Karnataka, India

## Abstract
We evaluated whether satellite-derived dynamics add information about CGWB groundwater over-exploitation status beyond location and terrain, using nested models (coordinates; + topography; + satellite dynamics) under buffered leave-one-taluk-out validation at 0-50 km. Complete imagery (2014-2023) was available for 44 over-exploited and 6 Safe taluks in Karnataka. At the pre-specified 25 km buffer, AUC was 0.485 for coordinates plus topography and 0.705 with satellite dynamics (ΔAUC = +0.220, 95% bootstrap CI -0.289 to +0.591). Intervals for the paired difference included zero at every buffer, and the satellite model's own intervals included chance, so the pilot is inconclusive. We also report a data audit showing that image missingness and patch count alone can separate the classes in an uncurated export, and the controls used to detect this.

---

## 1. Introduction
Groundwater over-exploitation is an escalating hydrogeological crisis across the hard-rock basalt and crystalline aquifers of Karnataka, India, where intensive borewell-fed irrigation withdraws groundwater substantially faster than seasonal monsoon recharge (Rodell et al., 2009; Famiglietti, 2014). The official statutory authority for groundwater classification in India is the Central Ground Water Board (CGWB), Ministry of Jal Shakti, which publishes periodic Dynamic Ground Water Resource Assessment (GWRA) reports. The CGWB classification is based on the "Stage of Extraction" (SOE %)—a mathematical accounting ledger computed as the ratio of annual groundwater draft to annual extractable recharge:
$$\text{SOE (\%)} = \left( \frac{\text{Annual Groundwater Draft}}{\text{Net Annual Extractable Groundwater Resource}} \right) \times 100$$
Administrative units with $\text{SOE} > 100\%$ are designated **Over-Exploited**, while units with $\text{SOE} < 70\%$ are designated **Safe** (CGWB, 2024).

In recent years, machine learning models applied to spaceborne remote sensing time-series have been proposed to detect regional aquifer depletion. However, spaceborne sensors (optical reflectance, thermal infrared emissions, microwave radar backscatter) observe only surface biophysical conditions, such as canopy vigor, soil moisture, and evapotranspirative cooling deficits, rather than subsurface hydrostatic heads or extraction accounting balances. Consequently, remote sensing models face acute risks of learning spurious correlations:
1. **Spatial Autocorrelation:** Adjacent administrative units share climate, cropping patterns, and soil conditions (Tobler, 1970). Without spatial buffering, machine learning models tend to memorize regional coordinates rather than real hydrological dynamics (Roberts et al., 2017; Ploton et al., 2020).
2. **Data Acquisition Artifacts:** Variations in cloud masking, sensor coverage boundaries, or incomplete raster exports can correlate with regional administrative boundaries, creating artificial classification shortcuts (Meyer et al., 2018).

In this study, we perform a forensic data audit on multi-sensor satellite imagery across Karnataka taluks and execute a nested feature benchmark under geographically buffered Leave-One-Taluk-Out (LOTO) cross-validation to assess whether multi-sensor satellite dynamics provide predictive information beyond spatial location and terrain.

---

## 2. Dataset and Data Quality Audit

### 2.1 Study Area and Ground Truth Labels
The study area focuses on administrative taluks across Karnataka, India. Ground truth classifications were derived from the Central Ground Water Board's statutory assessment (GWRA 2024/2025 assessment cycle). In this pilot cohort, 44 taluks are classified as Over-Exploited ($\text{Label} = 1$) and 6 taluks are classified as Safe ($\text{Label} = 0$), excluding intermediate semi-critical units to evaluate clean binary class separation.

### 2.2 Remote Sensing Multi-Sensor Channels
Satellite observations span the 10-year decadal record from 2014 to 2023 across 13 physical input channels:
1. **Optical Spectral Indices (Sentinel-2 MSI / Landsat 8):**
   * **NDVI (Normalized Difference Vegetation Index):** Canopy photosynthetic vigor and biomass.
   * **NDWI (Normalized Difference Water Index):** Surface canopy water content and open moisture.
   * **EVI (Enhanced Vegetation Index):** Canopy dynamics optimized for atmospheric resistance.
2. **Thermal & Climate Dynamics:**
   * **LST (Land Surface Temperature):** Derived from Landsat 8/9 TIRS and MODIS, capturing surface evaporative cooling deficits.
   * **Precipitation (CHIRPS):** Daily rainfall aggregated over the pre-monsoon and monsoon recharge seasons.
   * **Evapotranspiration (MODIS MOD16A2GF):** Canopy moisture flux and crop water consumption.
3. **Synthetic Aperture Radar (Sentinel-1 C-Band SAR):**
   * **VV and VH Polarizations:** Dielectric response to soil roughness and root-zone moisture.
   * **SAR Valid:** Binary indicator of valid radar data acquisition.
4. **Topographical Constraints (SRTM GL1 30m DEM):**
   * **Elevation, Slope, Aspect Sine, and Aspect Cosine:** Landscape drainage position and gravitational flow pathways.

### 2.3 Data Audit Findings & Quality Gate
A systematic data audit of the uncurated Earth Engine raster export revealed that image missingness (NaN fraction) alone achieved an AUC of 0.79, and spatial patch count alone achieved an AUC of 0.77 in separating Over-Exploited from Safe taluks. 

Our audit traced the root cause: **most Safe taluks were missing from the Earth Engine export (94 of 119 exclusions were unexported or static-only, and 25 were cloud/border masked)**. Unchecked, any machine learning model trained on uncurated exports would achieve high apparent performance by learning missingness patterns rather than surface hydrological signatures.

To eliminate this vulnerability, we implemented a data-quality gate:
* **Patch Sampling:** Each taluk is represented by 30 spatial patches sampled from the taluk's multi-spectral index (which includes both spatial grid points and observation well locations).
* **Missingness Filtering:** Missing pixel fractions were audited across all taluks, confirming maximum missingness of 5.56% and a mean under 1.5%.
* **Embedded Control Models:** We embedded two control models into the validation pipeline: `CTRL_missingness` (predicting from NaN fraction alone) and `CTRL_patch_count` (predicting from patch count).

After gating, complete multi-sensor time-series were available for 50 taluks (44 Over-Exploited and 6 Safe).

---

## 3. Methodology

### 3.1 Feature Representation & Aggregation
Each patch contains 10 annual multi-spectral observations ($10 \text{ years} \times 13 \text{ channels}$). Features were mechanically computed as 10-year temporal means and linear trend slopes across all 13 cached channels (26 features total). For static terrain channels (elevation, slope, aspect sine, aspect cosine), temporal slopes were nominally zero, producing 8 topography columns (means and zero slopes). Channel 8 contained a SAR acquisition validity mask, representing a potential missingness leak channel; however, the missingness control did not exceed chance (AUC 0.360–0.491).

Taluk locations were represented by bounding-box centroid coordinates converted to approximate kilometres from the boundary geometry, and pairwise distances between taluks were computed as bounding-box Euclidean distances in kilometres.

### 3.2 Nested Model Architecture
To isolate the incremental predictive contribution of each feature tier, we evaluate three nested models:
* **Model 0 ($M0$ - Coordinates Only):** Uses only projected centroid coordinates $(X, Y)$ to quantify pure geographic spatial autocorrelation.
* **Model 1 ($M1$ - Coordinates + Topography):** Combines coordinates with the 8 topographical terrain features.
* **Model 2 ($M2$ - Coordinates + Topography + Satellite Dynamics):** Combines coordinates, topography, and the complete 26 satellite dynamic features.

### 3.3 Spatially Buffered Leave-One-Taluk-Out (LOTO) Protocol
Standard random cross-validation leaks spatial proximity between neighboring taluks. We evaluate the models using buffered Leave-One-Taluk-Out (LOTO):
1. In each fold $i$, taluk $i$ is held out as the test instance.
2. All training taluks whose bounding boxes lie within a Euclidean buffer distance $b$ of taluk $i$ are purged from the training fold.
3. Feature imputation (median) and standardization (Z-score scaling) are fit strictly on the remaining training taluks.
4. An $L_2$-regularized logistic regression classifier ($C=0.1$, class-balanced weights) is trained and evaluated on the held-out taluk.
5. We evaluate five buffer distances: $b \in \{0, 15, 25, 35, 50\}\text{ km}$. The **25 km buffer** serves as the pre-specified primary contrast.

### 3.4 Statistical Uncertainty
Because rank-based metrics are sensitive when negative instances are scarce ($N_{\text{Safe}} = 6$), per-fold metrics are pooled, and 95% confidence intervals are evaluated using 2,000 paired bootstrap resamples. ΔAUC is computed on identical bootstrap resamples of taluks, applied to the pooled out-of-fold predictions.

---

## 4. Experimental Results

The frozen experimental results across all buffer exclusion distances are presented in Table 1.

### Table 1: Nested Model Performance Across Spatial Buffer Distances ($N = 50$: 44 Over-Exploited, 6 Safe)

| Buffer (km) | $N$ (OE / Safe) | $M0$ Coords (AUC) | $M1$ Coords+Topo (AUC) | $M2$ +Satellite (AUC) | $\Delta\text{AUC}$ ($M2 - M1$) | 95% CI ($\Delta\text{AUC}$) | `CTRL_missingness` (NaN AUC) | Scientific Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **0 km** | 50 (44 / 6) | 0.621 | 0.557 | 0.754 | +0.197 | [-0.236, +0.504] | 0.360 | Inconclusive (CI spans zero) |
| **15 km** | 50 (44 / 6) | 0.538 | 0.424 | 0.659 | +0.235 | [-0.245, +0.568] | 0.417 | Inconclusive (CI spans zero) |
| **25 km** | 50 (44 / 6) | 0.591 | 0.485 | 0.705 | +0.220 | [-0.289, +0.591] | 0.426 | Inconclusive (CI spans zero) |
| **35 km** | 50 (44 / 6) | 0.598 | 0.504 | 0.716 | +0.212 | [-0.310, +0.591] | 0.443 | Inconclusive (CI spans zero) |
| **50 km** | 50 (44 / 6) | 0.587 | 0.530 | 0.739 | +0.208 | [-0.292, +0.582] | 0.491 | Inconclusive (CI spans zero) |

### 4.1 Primary Contrast (25 km Buffer)
At the pre-specified 25 km buffer distance:
* Coordinates alone ($M0$) achieved an AUC of **0.591** (95% CI: [0.273, 0.894]).
* Coordinates plus topography ($M1$) achieved an AUC of **0.485** (95% CI: [0.189, 0.759]).
* Adding satellite dynamics ($M2$) yielded an AUC of **0.705** (95% CI: [0.375, 0.943]).
* The paired contrast yielded $\Delta\text{AUC} = +0.220$ with a 95% bootstrap confidence interval of **[-0.289, +0.591]**.

### 4.2 Critical Methodological Analysis
1. **Statistical Inconclusiveness:** While point estimates favored the satellite model ($M2$) across all buffer distances (+0.197 to +0.235 ΔAUC), the 95% bootstrap confidence intervals for the paired contrast span zero at every single buffer distance. Therefore, the satellite model cannot be statistically distinguished from the baseline on this dataset.
2. **Baseline Performance & Chance Overlap:** Coordinates plus topography ($M1$) performs at or below chance across all buffers (AUC 0.424 to 0.557), meaning that a substantial portion of the positive $\Delta\text{AUC}$ point estimate stems from $M1$ performing poorly. Crucially, the satellite model's ($M2$) own confidence intervals also include 0.50 (e.g., at 25 km: [0.375, 0.943]), indicating that the satellite model itself cannot be definitively confirmed to exceed chance.
3. **Integrity of Missingness Controls:** The missingness control model (`CTRL_missingness`) achieved AUCs between 0.360 and 0.491; the missingness control did not detectably differ from chance (AUC 0.36–0.49). With only 6 Safe taluks, neither reading is statistically distinguishable from 0.50, and we do not interpret it further. The patch-count control scored exactly 0.500 by construction, since every taluk contains exactly 30 patches.

---

## 5. Limitations and Future Work

### 5.1 Limitations
1. **Severe Negative Sample Scarcity:** Only 6 Safe taluks had complete, usable imagery in this raster export cohort. The 6 Safe taluks are simply the ones that happened to be exported, and they may not be representative of Safe taluks statewide. In a cohort of 6 negative units, misclassifying a single Safe taluk causes large swings in rank-based AUC, directly explaining the wide confidence intervals.
2. **Ground Truth Definition:** Official CGWB labels represent an administrative accounting calculation (draft versus recharge balancing) rather than a direct physical measurement of groundwater table depth. Spaceborne sensors cannot observe subsurface recharge equations directly.
3. **Vegetation Canopy Buffering:** In intensively irrigated agricultural taluks (such as Ajjampura), sustained surface irrigation can maintain green vegetative canopies despite severe deep-aquifer depletion, representing a working hypothesis that requires localized ground validation.
4. **Sample Neighborhood Representation:** Patches include observation-well locations, so taluk features describe the neighbourhood of sampled points, not the whole taluk.

### 5.2 Future Work
1. **Statewide Earth Engine Export:** Scale multi-sensor imagery extraction to all 147 administrative taluks in Karnataka to achieve a balanced statewide cohort with $\ge 50$ Safe taluks.
2. **Longitudinal Transition Modeling:** Rather than predicting static cross-sectional classifications, evaluate whether multi-annual satellite trend slopes can predict temporal status transitions across consecutive CGWB assessment cycles (e.g., identifying taluks transitioning from Safe to Over-Exploited over successive years).
3. **In-Situ Well Water Level Telemetry:** Validate patch-level satellite inferences directly against continuous piezometric borewell monitoring networks using spatially blocked cross-validation by well.

---

## 6. References
* Central Ground Water Board (CGWB) (2024). *National Compilation on Dynamic Ground Water Resources of India, 2023*. Ministry of Jal Shakti, Department of Water Resources, River Development & Ganga Rejuvenation, Government of India, New Delhi.
* Famiglietti, J. S. (2014). The global groundwater crisis. *Nature Climate Change*, 4(11), 945–948.
* Meyer, H., Reudenbach, C., Hengl, T., Katurji, M., & Nauss, T. (2018). Improving performance of spatio-temporal machine learning models using forward feature selection and target-oriented validation. *Environmental Modelling & Software*, 101, 1–9.
* Ploton, P., Mortier, F., Réjou-Méchain, M., Liang, J., Picard, N., Rossi, V., ... & Pélissier, R. (2020). Spatial validation reveals poor predictive performance of large-scale ecological mapping models. *Nature Communications*, 11(1), 4540.
* Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., ... & Dormann, C. F. (2017). Cross-validation strategies for data with spatial, temporal, or phylogenetic structure. *Ecography*, 40(8), 913–929.
* Rodell, M., Velicogna, I., & Famiglietti, J. S. (2009). Satellite-based estimates of groundwater depletion in India. *Nature*, 460(7258), 999–1002.
* Tobler, W. R. (1970). A computer movie simulating urban growth in the Detroit region. *Economic Geography*, 46(sup1), 234–240.
