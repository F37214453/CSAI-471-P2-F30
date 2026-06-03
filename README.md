# Discovering Iron Deficiency Using Smartphone-Based Tools

> **Reproducibility package** — preprocessing code, feature extraction, and model training for the paper:  
> *"Discovering Iron Deficiency Using Smartphone-Based Tools"*

---

## Overview

This repository contains the complete implementation of the signal processing pipeline and machine learning framework proposed in the paper. The system analyzes **capacitive thermal signals** captured from a mobile-compatible sensor array (34×16 grid) to classify subjects as **Normal** or **Iron Deficient (Low Iron)** in a fully non-invasive manner.

The pipeline covers three stages:

1. **Data Preprocessing & Feature Extraction** — multi-stage signal normalization and extraction of 35+ spatial and statistical features per frame
2. **Exploratory Data Analysis & Feature Evaluation** — statistical ranking of features using Cohen's d and Welch's t-test
3. **Model Training & Evaluation** — six classifiers trained under strict Leave-One-Subject-Out (LOSO) and GroupKFold cross-validation, with patient-level majority voting

---

## Repository Structure

```
├── Final_Preprocessing___Models_Training_Code.ipynb   # Main notebook (all sections)
└── README.md
```

---

## Methodology

### Section 1 — Data Preprocessing & Feature Extraction

- Sensor grid configuration: 34×16 pixels with a circular active mask (radius = 6), yielding **N active pixels**
- Three-stage preprocessing pipeline applied per frame:
  1. **Energy Normalization** — clips negative values and normalizes the frame so pixel values sum to 1
  2. **Whitening Transform** — applies covariance-based spatial equalization to remove inter-pixel correlations
  3. **Z-Score Normalization** — standardizes the whitened map and re-normalizes to a probability distribution
- **35+ features** extracted per frame across six categories:

| Category | Features |
|---|---|
| Signal Statistics | mean, std, variance, power, RMS, peak amplitude, crest factor |
| Distribution Shape | kurtosis, skewness, coefficient of variation |
| Spatial Entropy | spatial entropy, normalized entropy |
| Energy Concentration | top-10/25/50% energy, Gini coefficient |
| Radial Ring Energy | ring1–ring5 energy, inner/outer ratio, ring decay ratio |
| Zone & Radial Decay | core/mid/boundary energy, radial slope, curvature, spread, variance, kurtosis |

- Parallel processing via `joblib` across all subject CSV files
- Subject identity parsed from file names using regex; verified against expected participant count

### Section 2 — Exploratory Data Analysis & Feature Evaluation

- Side-by-side **boxplots** for all features comparing Normal vs. Low Iron distributions
- **Grouped bar charts** (8 feature groups, Min-Max normalized) with standard error bars
- **Statistical ranking** per feature:
  - Welch's t-test (unequal variance) for significance testing
  - Cohen's d for effect size estimation
  - Features labeled: 🌟 Excellent (|d| ≥ 0.8) / 👍 Good (|d| ≥ 0.5) / 🤏 Weak / ❌ Non-significant

### Section 3 — Model Training & Evaluation

#### 3.1 Baseline Comparison (LOSO)
Six classifiers evaluated under **Leave-One-Subject-Out** cross-validation:
`Logistic Regression` · `Random Forest` · `Extra Trees` · `Gradient Boosting` · `SVM` · `KNN`

Metrics reported at the **frame level**: Accuracy, Precision, Recall, F1-Score.

#### 3.2 Advanced Feature Selection & Hyperparameter Tuning
Greedy forward selection combined with hyperparameter search, optimizing **patient-level F1** via majority voting across frames. Four algorithms explored:

| Algorithm | Search Space |
|---|---|
| Logistic Regression | C ∈ {0.00001, 0.0001, 0.001, 0.01} |
| HistGradientBoosting | max_iter ∈ {50, 100} |
| Decision Tree | max_depth ∈ {3, 5, 7, None} |
| KNN | n_neighbors ∈ {3, 5, 7, 9} |

- Maximum features selected: **3** (to prevent overfitting on small cohort)
- Cross-validation: **GroupKFold (5 folds)** — groups defined at subject level to prevent data leakage
- Results reported at both **frame level** and **patient level** with confusion matrices

---

## Validation Protocol

To ensure **leakage-free** evaluation across all experiments:

- Subject identity is strictly enforced at the group level; no frames from the same subject appear in both train and test sets simultaneously
- Patient-level diagnosis is determined by **majority voting** across all frames of a subject
- `zero_division=0` enforced in all sklearn metrics calls
- Subjects with no test-fold coverage are flagged with explicit warnings

---

## Requirements

```
Python >= 3.8
pandas
numpy
scipy
scikit-learn
matplotlib
seaborn
joblib
```

Install dependencies:
```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn joblib
```

---

## Data

Raw data files are CSV-formatted thermal sensor recordings, organized per subject with condition labels encoded in file names (`N` = Normal, `L` = Low Iron). Each row corresponds to one frame containing a flattened 34×16 = 544-value delta signal.

> **Note:** Raw data is not included in this repository due to participant privacy constraints. The preprocessing and feature extraction code is fully reproducible given data in the expected format.

---

## Reproducibility

1. Clone this repository
2. Mount your data directory (Google Drive or local) and update `folder_path` in Cell 4
3. Run all cells sequentially in `Final_Preprocessing___Models_Training_Code.ipynb`

All random seeds are fixed (`random_state=42`) across all classifiers and splits.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{,
  title   = {Discovering Iron Deficiency Using Smartphone-Based Tools},
  author  = {},
  journal = {},
  year    = {2025}
}
```

---

## License

This code is released for academic and research purposes only.
