# HW-2: Gradient Features & Aggregation (Fisher Vectors)

**Course:** ECE 479/5582 — Computer Vision  
**Dataset:** NWPU-RESISC45 (first 15 classes × 60 imgs/class = 900)  
**Features:** SIFT, Dense SIFT, HoGf (grayscale only)  
**Aggregation:** PCA → GMM (diag) → Fisher Vector; 1-NN, true Leave-One-Out

> Assignment summary and required functions/tables are from the HW-2 prompt. :contentReference[oaicite:0]{index=0}  
> FV reference implementation ideas adapted from `fisher.py`. 

---

## Environment

```bash
python3 -m venv .venv-hw2
source .venv-hw2/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name hw2-cv --display-name "Python (hw2-cv)"
Dataset

Download NWPU-RESISC45 and point the notebook to your local path.
We use the first 15 alphabetically named classes and 60 images per class.

Example path used during development:
/home/manny-buff/projects/cv-hw1/data/NWPU-RESISC45
How to run

Launch Jupyter and open notebooks/HW2_Features_Aggregation.ipynb:
jupyter notebook notebooks
Execute cells in order. The notebook:

Implements getImageFeatures(im, opt) for 'sift', 'dsift', 'hogf'

Extracts & saves f_sift{} and f_dsift{} to reports/gradient_features.mat

Fits PCA (kd ∈ {16, 32}) and GMM (nc ∈ {32, 64, 128})

Builds Fisher Vectors with power + L2 norm

Evaluates true LOO 1-NN (excludes self matches) and outputs accuracy tables & confusion maps
Outputs (in reports/)

accuracy_sift_fv_trueLOO.csv

accuracy_dsift_fv_trueLOO.csv

accuracy_hogf_fv_trueLOO.csv

cm_*_fv_kd{16|32}_nc{32|64|128}.png (18 figures)

gradient_features.mat (HDF5 v7.3; SIFT & DSIFT cell arrays, labels, class names)

Best configs (true LOO on 15×60 subset):

SIFT FV: kd=16, nc=32, acc ≈ 0.201

DSIFT FV: kd=32, nc=128, acc ≈ 0.559

HoGf FV: kd=16, nc=32, acc ≈ 0.340

Dense SIFT + FV clearly outperforms sparse SIFT and HoGf on this subset.
Notes

We standardize (z-score) PCA outputs before GMM on HoGf and increase reg_covar if EM collapses.

Data (data/), raw features (features/), and models (models/) are ignored by Git.
