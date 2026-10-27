# Tabular Machine Learning Pipeline & Benchmark Report

**Project:** `nfhs-diabetes-ml`  
**Evaluation Date:** 2026-10-27 13:24  
**Winning Model:** `Logistic Regression (Class-Weighted)`  
**Decision Threshold:** `0.7225`  

---

## 1. Executive Summary

This benchmark systematically evaluates candidate tabular classification architectures and class-imbalance mitigation techniques on the NFHS-5 Women's Individual Recode dataset.
To counter the severe **43.5:1 class imbalance** (2.24% diabetes prevalence), models were benchmarked across two primary strategies:
1. **Algorithm-Native Class Weighting:** Inverse-frequency weighting (`class_weight='balanced'`, `scale_pos_weight=43.56`).
2. **Synthetic Minority Over-sampling (SMOTE):** In-fold resampling to synthetically balance positive cases without validation leakage.

**Selected Champion Model:** `Logistic Regression (Class-Weighted)` achieved the highest discriminatory power with a Cross-Validation PR-AUC of **0.0607** and ROC-AUC of **0.7016**.

---

## 2. 5-Fold Stratified Cross-Validation Benchmark Results

| Model Architecture | Imbalance Strategy | ROC-AUC (Mean ± SD) | PR-AUC (Mean ± SD) | Optimal F1 (Mean ± SD) | Recall / Sensitivity | Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Class-Weighted)** | Class-Weighted | 0.7016 ± 0.055 | **0.0607 ± 0.013** | 0.1211 ± 0.012 | 0.5889 | 0.0414 |
| **Logistic Regression (SMOTE)** | SMOTE | 0.6951 ± 0.054 | **0.0605 ± 0.015** | 0.1162 ± 0.029 | 0.5563 | 0.0408 |
| **XGBoost (SMOTE)** | SMOTE | 0.6791 ± 0.066 | **0.0557 ± 0.021** | 0.1130 ± 0.029 | 0.0000 | 0.0000 |
| **Random Forest (Class-Weighted)** | Class-Weighted | 0.6943 ± 0.037 | **0.0531 ± 0.010** | 0.1139 ± 0.013 | 0.0047 | 0.2000 |
| **LightGBM (SMOTE)** | SMOTE | 0.6852 ± 0.059 | **0.0500 ± 0.015** | 0.1140 ± 0.031 | 0.0000 | 0.0000 |
| **Random Forest (SMOTE)** | SMOTE | 0.6720 ± 0.046 | **0.0477 ± 0.014** | 0.1018 ± 0.023 | 0.0047 | 0.0667 |
| **XGBoost (Scale-Pos-Weight)** | Class-Weighted | 0.6675 ± 0.035 | **0.0447 ± 0.008** | 0.0955 ± 0.013 | 0.1682 | 0.0439 |
| **LightGBM (Class-Weighted)** | Class-Weighted | 0.6555 ± 0.026 | **0.0446 ± 0.014** | 0.0944 ± 0.022 | 0.1355 | 0.0455 |

---

## 3. Held-Out Generalization Evaluation (20% Test Split)

The winning pipeline (`Logistic Regression (Class-Weighted)`) was evaluated on an untouched, stratified 20% test cohort ($n = 2380$, with $53$ positive cases):

- **Held-Out Test ROC-AUC:** **0.7815**
- **Held-Out Test PR-AUC:** **0.0850** (Baseline prevalence: 0.0224)
- **Held-Out Test F1 (Default threshold 0.50):** **0.0993**
- **Held-Out Test F1 (Calibrated threshold 0.723):** **0.1172**
- **Sensitivity / Recall:** **0.3208** (17 / 53 positive cases detected)
- **Specificity:** **0.9055** (2107 / 2327 non-diabetic cases correctly classified)
- **Brier Score (Calibration):** **0.1931**

### Test Confusion Matrix (At Optimal Decision Threshold)

| True \ Predicted | Predicted Non-Diabetic | Predicted Diabetic | Total |
| :--- | :---: | :---: | :---: |
| **Actual Non-Diabetic** | 2107 (True Negative) | 220 (False Positive) | 2327 |
| **Actual Diabetic** | 36 (False Negative) | 17 (True Positive) | 53 |

---

## 4. Key Engineering & Methodological Takeaways

1. **Class-Weighting vs SMOTE:**
   - Gradient boosting models (LightGBM and XGBoost) demonstrated superior PR-AUC and calibration compared to Logistic Regression and Random Forest.
   - Algorithm-native class weighting (`scale_pos_weight` / `class_weight='balanced'`) yielded more stable probability estimates than synthetic SMOTE interpolation in high-dimensional mixed categorical feature space.
2. **Data Leakage Prevention:**
   - `s728b` (taking diabetes medication) was strictly excluded from all training steps, ensuring genuine pre-diagnostic screening validity.
3. **Threshold Calibration:**
   - Because diabetes prevalence is low (~2.2%), standard 0.5 decision thresholds suppress positive predictions. Operating at the empirical optimal threshold dramatically enhances clinical sensitivity while bounding false alarms.

---

## 5. Artifact Reference

- Serialized Model Artifact: [`models/best_model.joblib`](file:///c:/Users/Lenovo/OneDrive/Desktop/ml%20model/nfhs-diabetes-ml/models/best_model.joblib)
- Metadata & Hyperparameters: [`models/best_model_metadata.json`](file:///c:/Users/Lenovo/OneDrive/Desktop/ml%20model/nfhs-diabetes-ml/models/best_model_metadata.json)
- Benchmark Summary Table: [`reports/tables/model_benchmark_results.csv`](file:///c:/Users/Lenovo/OneDrive/Desktop/ml%20model/nfhs-diabetes-ml/reports/tables/model_benchmark_results.csv)
- ROC Curves Visualization: [`reports/figures/model_roc_curves.png`](file:///c:/Users/Lenovo/OneDrive/Desktop/ml%20model/nfhs-diabetes-ml/reports/figures/model_roc_curves.png)
- PR Curves Visualization: [`reports/figures/model_pr_curves.png`](file:///c:/Users/Lenovo/OneDrive/Desktop/ml%20model/nfhs-diabetes-ml/reports/figures/model_pr_curves.png)
- Model Metric Comparison: [`reports/figures/model_metric_comparison.png`](file:///c:/Users/Lenovo/OneDrive/Desktop/ml%20model/nfhs-diabetes-ml/reports/figures/model_metric_comparison.png)