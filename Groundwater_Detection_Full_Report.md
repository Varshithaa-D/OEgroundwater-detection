# Comprehensive Project Report: Deep Learning Framework for Regional Groundwater Over-Exploitation Detection Using Multi-Sensor Satellite Imagery

## Abstract
Groundwater depletion is a critical global crisis, historically monitored through expensive, sparse, and manually collected telemetry well data. This project introduces a scalable, automated framework that utilizes multi-sensor satellite imagery and deep learning to detect underground water stress from surface-level indicators. By fusing optical, radar, climatic, and topographical data, the system successfully classifies over-exploited geographic regions with exceptionally high accuracy, culminating in an interactive geographical dashboard for policy intervention.

---

## 1. Study Area and Scalability
While the framework was fundamentally designed as a universally scalable, region-agnostic architecture capable of being deployed globally, the primary training and validation were conducted across **44 critically over-exploited Taluks**. 
* **Agro-Climatic Diversity:** The selected regions encompass diverse geographical and agro-climatic zones (including arid, semi-arid, and transition zones). Testing the model across diverse terrains ensures that the neural network learns generalized hydrogeological features rather than memorizing localized geographic artifacts. 
* **Grid Mapping:** The entire study area was divided into highly granular, uniform spatial grid blocks (approx. 1km x 1km), allowing for precise, localized predictions rather than broad, inaccurate regional estimates.

---

## 2. Dataset Collection and Feature Engineering
The core innovation of this project lies in the fusion of heterogeneous datasets. Ground truth data was merged with a massive 12-channel satellite imagery stack.

### A. Remote Sensing (Satellite) Input Features (12 Channels)
The model ingests high-resolution satellite Data Cubes consisting of the following environmental indicators:
1. **Optical Indices (Sentinel-2 Multispectral):** 
   * **NDVI (Normalized Difference Vegetation Index):** Measures vegetation health. Stressed vegetation is a primary proxy for shallow groundwater depletion.
   * **NDWI (Normalized Difference Water Index):** Highlights surface water bodies and canopy moisture.
   * **EVI (Enhanced Vegetation Index):** Corrects for atmospheric conditions and canopy background noise.
2. **Synthetic Aperture Radar (Sentinel-1 SAR):** 
   * **VV and VH Polarizations:** Unlike optical sensors, radar penetrates cloud cover. It actively measures surface roughness, soil moisture content, and structural changes in land cover that correlate with underground water stress.
3. **Climatic Variables (CHIRPS & MODIS):**
   * **Rainfall (mm):** Historical and seasonal precipitation anomalies.
   * **Evapotranspiration (ET):** The transfer of water from the land to the atmosphere, a key component of the water cycle.
   * **Land Surface Temperature (LST):** Areas with depleted groundwater typically exhibit higher surface temperatures due to reduced evaporative cooling.
4. **Topographical Data (Digital Elevation Model - DEM):**
   * **Elevation, Slope, and Aspect (Sine/Cosine):** Water naturally accumulates in valleys and runs off steep slopes. Providing topography allows the neural network to understand the physical constraints of the watershed.

### B. Ground Truth Values (Target Variables)
The target labels for supervised learning were established using historical government groundwater telemetry well data.
* **GWL (Groundwater Level):** Time-series data of underground water depths.
* **Labeling Strategy:** Wells showing a sustained, critical decline in water levels beyond safe recharge thresholds were mathematically classified as **Stressed / Over-Exploited (`Label = 1`)**. Wells with stable or rising water tables were classified as **Safe (`Label = 0`)**.

---

## 3. Advanced Data Preprocessing
Raw satellite data is inherently noisy and geographically misaligned. The following preprocessing pipeline was executed prior to model training:
* **Spatial Alignment (Reprojection):** All TIFF files and vectors were projected into a unified Coordinate Reference System (CRS) using PyProj, ensuring perfect pixel-to-pixel alignment across all 12 channels.
* **Cloud Masking and Compositing:** Dense monsoon cloud cover frequently obscures optical sensors like Sentinel-2. To bypass this, we generated temporal median composites (e.g., Year-to-Date medians), filtering out cloud shadows and atmospheric noise to produce a pristine, cloud-free representation of the grid.
* **Zarr Array Compression:** The massive multidimensional datasets were chunked and compressed into a high-throughput spatial database (`hydrosentry.zarr`) to prevent memory overflow (OOM) errors during Deep Learning training.
* **Feature Normalization:** All channels were statistically normalized (min-max and z-score scaling) so that variables with large ranges (like Elevation) would not mathematically dominate variables with small ranges (like NDVI) during backpropagation.

---

## 4. Deep Learning Model Architecture
To process the complex spatial relationships within the satellite arrays, we utilized a Deep Convolutional Neural Network (CNN), specifically a modified **ResNet-18** architecture.

* **Why ResNet-18?** Standard deep networks suffer from the "vanishing gradient" problem, where the model stops learning in its deepest layers. ResNet (Residual Network) introduces "skip connections" that allow gradients to flow freely, enabling deep, stable learning. 
* **Input Layer Modification:** A standard ResNet is designed for 3-channel (RGB) images. We re-engineered the input convolutional layer to accept our **12-channel spatial arrays**.
* **Feature Extraction:** The convolutional layers act as hierarchical feature extractors. Early layers detect basic edges (e.g., rivers, slopes), while deeper layers learn complex non-linear correlations (e.g., "If rainfall is low AND elevation is high AND vegetation is stressed, the probability of groundwater depletion is critical").

---

## 5. Training Process and Validation Methodology
Training a spatial model requires rigorous validation to ensure it performs accurately in the real world.
* **Spatial Block Cross-Validation:** A standard random train/test split fails in remote sensing due to "Tobler's First Law of Geography" (pixels close to each other are highly correlated). If adjacent pixels leak into both the train and test sets, the model will falsely report high accuracy. We solved this using **Spatial Block Splitting**. The geographic map was divided into massive macro-blocks. The model was trained on one set of blocks and forced to predict on entirely unseen geographic blocks, proving true generalization.
* **Optimization:** The network was trained using categorical cross-entropy loss, utilizing modern optimizers (e.g., Adam/SGD) with dynamic learning rate scheduling to reach the global minimum loss.

---

## 6. Final Results and Model Performance
The customized ResNet-18 model achieved exceptional, state-of-the-art results on the isolated spatial test set. The metrics validate the hypothesis that fused surface-level satellite data can accurately predict subsurface groundwater depletion.

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **94.6%** | The model correctly classifies safe vs. over-exploited grids 94.6% of the time. |
| **AUC** | **0.992** | An Area Under the ROC Curve of 0.992 indicates near-perfect mathematical separability between the classes. |
| **F1-Macro** | **0.941** | The harmonic mean of Precision and Recall, proving the model handles both classes fairly without bias. |
| **Precision** | **93.2%** | When the model predicts a grid is over-exploited, it is correct 93.2% of the time. |
| **Recall** | **95.3%** | The model successfully identifies 95.3% of all truly over-exploited regions (minimizing dangerous false negatives). |

---

## 7. Deployment: Output Fetching and Visualization Dashboard
Deploying heavy deep learning models for live end-user inference presents a significant computational challenge. Running a 12-channel ResNet-18 model over a massive geographic region requires expensive Cloud GPU infrastructure and roughly 30 minutes of processing time per Taluk.

To bridge the gap between Data Science and practical government application, we designed a **Digital Twin Executive Dashboard**:
1. **Offline Inference Caching:** The PyTorch model is executed entirely in the backend pipeline. Its exact predictions—including the specific floating-point probabilities (e.g., 98.6% stress) and real-world GPS coordinates (Lat/Lon) for every grid cell—are cached into an optimized `.parquet` database (`cells_wellhead.parquet`).
2. **Desktop Visualization GUI:** We engineered a localized Tkinter desktop application that acts as the frontend client. The app queries the pre-computed database and the `hydrosentry.zarr` spatial imagery, instantly rendering Regional Heatmaps overlaying the predicted groundwater stress onto the baseline satellite view.
3. **Interactive Web GIS (Folium):** To make the intelligence actionable for field intervention, the system exports the critical danger zones to a dynamic HTML Web Map. By integrating with the Google Maps API, it completely bypasses standard open-source rendering limits, allowing authorities to securely pan, zoom, and inspect the specific street and village-level geographies of the predicted over-exploited blocks. 

This end-to-end architecture seamlessly translates raw, multi-sensor space data into actionable, highly accurate hydrogeological intelligence.
