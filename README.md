# Forest Cover Classification using Remote Sensing and Deep Learning

<p align="center">
  <img src="images/Forest Classification.png" height="200px" />
  <img src="images/Primary forest area.png" height="200px" />
</p>

This project implements a deep learning model in PyTorch to classify forest cover types using multi-source satellite data — Landsat, ALOS PALSAR, and GEDI. The goal is to classify regions into **Primary Forest**, **Secondary Forest**, and **Others** based on tree cover and canopy height predictions.

---

## 📦 Data Sources

- **Landsat 8/9** (Optical imagery)
- **ALOS PALSAR** (Radar Imagery)
- **GEDI L2B** (Canopy height and tree cover from GEDI LiDAR)

All satellite data were preprocessed into consistent spatial resolution and aligned using a common grid system (e.g., 10m or 30m).

---

## 🧠 Model Overview

A simple  **regression neural network** is trained to predict two outputs:

1. **Canopy Height** (in meters, clipped between 0 and 40m)
2. **Tree Cover percentage** (normalized between 0 and 1)

These are later used to classify forest cover types.

---

## 🏷️ Forest Classification Logic

| Tree Cover | Canopy Height | Class            |
|------------|----------------|------------------|
| > 0.7      | > 20m          | Primary Forest   |
| 0.4–0.7    | 5–20m          | Secondary Forest |
| < 0.4      | < 5m           | Others           |

These thresholds can be adjusted based on region-specific forest structure.

---

## 🔧 Preprocessing Pipeline

1. **Load satellite data** from data_preprocessor and data_accesor module
2. **Load GEDI data** get_gedi_excel from data_preprocessor module
3. **Normalize features** using `StandardScaler`
4. **Load model and predict** canopy height and tree cover
5. **Classify forest type** using defined logic
6. **Visualize and export maps** using `matplotlib` and `xarray`

---

- Loss: MSELoss,
- Metric: R² score,
- Optimizer: Adam

---



