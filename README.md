# 🌍 Groundwater Over-Exploitation Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red)
![GIS](https://img.shields.io/badge/GIS-Spatial_Analysis-green)
![Accuracy](https://img.shields.io/badge/Accuracy-94.6%25-brightgreen)

A highly scalable, Deep Learning-based geographic information system (GIS) that fuses multi-sensor satellite imagery to accurately detect and map underground water stress and over-exploited regional aquifers.

## 📌 Project Overview
Monitoring groundwater depletion traditionally relies on expensive, sparse telemetry well data. This project bridges the gap between deep learning and remote sensing by using a customized **ResNet-18 Neural Network** to analyze 12 channels of surface-level satellite data and predict subsurface water stress. 

The framework is deployed via an interactive **Executive Desktop Dashboard**, allowing policymakers to instantly visualize over-exploited zones on interactive geographic maps without needing heavy cloud-compute during field operations.

### Key Features:
* **Multi-Sensor Fusion (12-Channel):** Integrates Sentinel-2 (Optical Indices: NDVI, NDWI, EVI), Sentinel-1 (SAR Radar: VV, VH), Topography (DEM: Elevation, Slope, Aspect), and Climate variables (CHIRPS Rainfall, ET, LST).
* **Deep Learning Engine:** A customized PyTorch ResNet-18 architecture utilizing Spatial Block Cross-Validation to ensure true geographic generalization.
* **Offline Digital Twin:** The heavy AI inference is cached into a high-speed spatial database (`cells_wellhead.parquet`), allowing instant dashboard rendering on standard consumer hardware.
* **Interactive Web GIS:** Automatically exports critical danger zones to interactive Folium Web Maps using the Google Maps API for precise, street-level intervention planning.

---

## 📊 Model Performance
The architecture achieved state-of-the-art results on isolated spatial test blocks:
* **Accuracy:** 94.6%
* **AUC (Area Under Curve):** 0.992
* **F1-Macro Score:** 0.941
* **Precision:** 93.2%
* **Recall:** 95.3%

*(For a comprehensive breakdown of the methodology, architecture, and spatial dataset, please refer to the `Groundwater_Detection_Full_Report.md` included in this repository).*

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/groundwater-detection.git
   cd groundwater-detection
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Requirements:**
   *Note: Due to GitHub file size limits, the massive spatial datasets (`hydrosentry.zarr`, `index.parquet`, and model weights inside `runs/`) are not included in this repository. Ensure you have the localized database files in your root directory before running the dashboard.*

---

## 🚀 Usage

To launch the Executive Spatial Dashboard, run the following command in your terminal:

```bash
python desktop_app.py
```

### Dashboard Workflow:
1. **Region Selection:** Select a target Taluk from the dropdown menu.
2. **Detect Baseline Areas:** Click the detection button to query the deep learning cache and generate the regional heatmap.
3. **Export Web Map:** Click the Export button to automatically generate an interactive HTML map and open it in your default web browser for granular inspection.

---

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch, Torchvision
* **Data Processing:** Pandas, NumPy, Zarr, PyProj
* **Visualization:** Tkinter (GUI), Matplotlib, Folium (Web Maps)
* **Remote Sensing:** Sentinel-1, Sentinel-2, CHIRPS, SRTM DEM
