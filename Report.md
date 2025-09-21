# HW-2 Report: Gradient Features and Aggregation

**Course:** ECE 479/5582 Computer Vision  
**Student:** *Your Name*  
**Repo:** [CS5582-HW-2](https://github.com/BuffaloManwich/CS5582-HW-2)

---

## 1. Dataset
- NWPU-RESISC45 aerial dataset (≈45 classes × 700 images).  
- For HW-2: selected the first 15 classes (airplane … freeway), 60 images each = **900 images**.  
- Converted to grayscale for feature extraction.

---
## 2. Feature Extraction
Implemented `getImageFeatures(im, opt)` with three options:
- **SIFT** — classic 128-dim descriptors at keypoints.  
- **Dense SIFT (DSIFT)** — descriptors sampled on an 8px grid.  
- **HoGf** — Histogram of Oriented Gradients features (~34k dim per image).

Saved:
- `f_sift{}`, `f_dsift{}` as cell arrays in `reports/gradient_features.mat`.  
- Labels and class names included.

---

## 3. PCA + GMM
- PCA fitted to reduce descriptors to **kd ∈ {16, 32}**.  
- GMMs trained with **nc ∈ {32, 64, 128}** diagonal components.  
- Stored `{mean, var, prior}` in `models/gmm/*.pkl`.  
- Eigenvalue spectra plotted for SIFT vs. DSIFT.

---

## 4. Fisher Vector Aggregation
- Implemented `getFisherVector(f, A, gmm, kd, nc)`.  
- Stats: 0th, 1st, 2nd order per Gaussian.  
- Normalization: signed square root + L2 norm.  
- Verified FV length = `nc*(1+kd+kd)`.

---

## 5. Evaluation
- Classifier: 1-NN with **true leave-one-out** (self excluded).  
- Confusion matrices (18 total) saved in `reports/`.

### 5.1 Accuracy Tables (TRUE LOO)

**SIFT FV**

| kd/nc | 32   | 64   | 128  |
|-------|------|------|------|
| 16    | 0.201 | 0.138 | 0.090 |
| 32    | 0.159 | 0.103 | 0.076 |

**DSIFT FV**

| kd/nc | 32   | 64   | 128  |
|-------|------|------|------|
| 16    | 0.474 | 0.488 | 0.484 |
| 32    | 0.519 | 0.523 | 0.559 |

**HoGf FV**

| kd/nc | 32   | 64   | 128  |
|-------|------|------|------|
| 16    | 0.340 | 0.303 | 0.229 |
| 32    | 0.329 | 0.289 | 0.270 |

---

## 6. Findings
- **Dense SIFT + FV** is clearly strongest (~56%).  
- **SIFT + FV** trails badly (≤20%).  
- **HoGf + FV** competitive but below DSIFT.  
- Larger nc can hurt (esp. SIFT, HoGf) due to over-fragmentation.

---
## 7. Deliverables
- Notebook: `notebooks/HW2_Features_Aggregation.ipynb`  
- Report: `Report.md` (this file)  
- Outputs: in `reports/` (CSVs + confusion PNGs)  
- Repo: [github.com/BuffaloManwich/CS5582-HW-2](https://github.com/BuffaloManwich/CS5582-HW-2)

