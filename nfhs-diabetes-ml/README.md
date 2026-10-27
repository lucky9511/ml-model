# NFHS-5 Diabetes Machine Learning (`nfhs-diabetes-ml`)

An end-to-end, reproducible, and explainable tabular machine learning system to classify diabetes status using **NFHS-5 India (2019–21) Women's Individual Recode** data.

---

## 📌 Project Overview

The National Family Health Survey 5 (NFHS-5) provides extensive demographic, socioeconomic, anthropometric, and biomarker data across India. This project utilizes the **Women's Individual Recode (IAIR7D)** dataset to develop interpretable machine-learning classifiers for diabetes identification and risk stratification.

### Key Objectives:
- **Tabular ML Pipelines**: Build and benchmark multiple tabular models (Logistic Regression, Random Forest, LightGBM, XGBoost).
- **Class Imbalance Management**: Address survey imbalance using techniques like SMOTE and class-weight balancing.
- **Explainable AI (XAI)**: Leverage SHAP (SHapley Additive exPlanations) for local and global model interpretability.
- **Reproducibility**: Enforce deterministic operations across the pipeline via `random_state=42`.
- **Interactive App**: Deploy an inference and risk-scoring dashboard with Streamlit.

---

## 📁 Project Directory Structure

```text
nfhs-diabetes-ml/
├── data/
│   ├── raw/                # Unmodified NFHS-5 raw files (e.g., .dta, .sav)
│   ├── interim/            # Intermediate transformed datasets
│   └── processed/          # Final feature-engineered datasets ready for modeling
├── notebooks/              # Jupyter notebooks for EDA and prototyping
├── src/                    # Source code package
│   ├── __init__.py         # Package initializer
│   ├── config.py           # Configuration loader, paths, and constants (random_state=42)
│   └── utils.py            # Reproducibility seed setup, logging, and artifact helpers
├── app/                    # Streamlit interactive application
├── models/                 # Serialized model artifacts (.joblib)
├── reports/
│   ├── figures/            # Generated EDA and SHAP visualizations
│   └── tables/             # Model metrics, confusion matrices, and feature importance tables
├── tests/                  # Unit and integration test suite
├── .gitignore              # Git ignore rules for data, models, and virtualenvs
├── config.yaml             # Central configuration file
├── README.md               # Project documentation
└── requirements.txt        # Python package dependencies
```

---

## ⚙️ Setup and Installation

### 1. Prerequisites
- Python 3.10+ installed
- Windows / Linux / macOS

### 2. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🎯 Reproducibility

To guarantee deterministic, reproducible results, a global random seed is maintained:
```python
RANDOM_STATE = 42
```
All random number generators (`numpy`, `random`, `scikit-learn`, `xgboost`, `lightgbm`, etc.) are configured to use this seed.

---

## 🚀 Next Steps

1. **Data Ingestion**: Place raw NFHS-5 Women's Recode data files into `data/raw/`.
2. **Exploratory Data Analysis (EDA)**: Inspect missingness, biomarkers, and variable distributions in `notebooks/`.
3. **Data Preprocessing & Feature Engineering**: Implement target definitions and categorical encodings under `src/`.
4. **Model Training & Evaluation**: Train baseline and ensemble models with cross-validation.
5. **SHAP Interpretation**: Generate summary plots and local risk waterfall plots.
6. **Application**: Launch the Streamlit dashboard for real-time risk assessment.
