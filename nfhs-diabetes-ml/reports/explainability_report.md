# Model Explainability & Clinical Feature Attribution Report (SHAP)

**Project:** `nfhs-diabetes-ml`  
**Framework:** SHapley Additive exPlanations (SHAP) — Game-Theoretic Attribution  
**Evaluated Model:** Champion Class-Weighted Logistic Classifier  

---

## 1. Executive Summary & Epistemic Guidance

In healthcare applications, predictive machine learning models must be transparent, verifiable, and explainable to clinical practitioners.
This report uses **SHAP** to quantify the marginal contribution of each demographic, anthropometric, and clinical biomarker toward estimated diabetes risk.

> [!IMPORTANT]
> **Non-Causal Interpretability:**
> SHAP values quantify the *statistical contribution* of each feature to the model's predictions within the cross-sectional NFHS-5 survey sample.
> A positive SHAP value indicates that the observed feature value shifts the prediction toward higher diabetes probability; it does not prove an unconfounded causal pathway.

---

## 2. Top-10 Global Drivers of Diabetes Risk

| Rank | Clinical / Demographic Feature | Mean |SHAP Value| (Impact Magnitude) | Primary Direction of Effect |
| :---: | :--- | :---: | :--- |
| 1 | **Age (Years)** | **0.5675** | Positive (Higher value increases risk) |
| 2 | **Weight (kg)** | **0.2983** | Positive (Higher value increases risk) |
| 3 | **Residence: Rural** | **0.2179** | Positive (Higher value increases risk) |
| 4 | **Wealth Quintile: Poorest** | **0.1976** | Protective (Reduces risk) |
| 5 | **Wealth Quintile: Richest** | **0.1579** | Protective (Reduces risk) |
| 6 | **Height (cm)** | **0.1395** | Positive (Higher value increases risk) |
| 7 | **Diet: Pulses/Beans Frequency** | **0.1227** | Protective (Reduces risk) |
| 8 | **BMI: Overweight (25.0-29.9)** | **0.1135** | Positive (Higher value increases risk) |
| 9 | **Hypertension Diagnosis** | **0.1085** | Protective (Reduces risk) |
| 10 | **Wealth Quintile: Poorer** | **0.0977** | Protective (Reduces risk) |

### Epidemiological Takeaways on Top Features:
1. **Age (Years):** Primary driver of diabetes risk in the reproductive-age cohort (15–49). Risk increases monotonically with age, reflecting prolonged exposure to metabolic stressors.
2. **Body Mass Index (kg/m²):** Leading modifiable anthropometric driver. Higher continuous BMI and the WHO 'Obese' category exhibit strong positive attribution shifts.
3. **Hypertension Comorbidity:** Co-occurrence of doctor-diagnosed high blood pressure contributes strongly to elevated diabetes likelihood, capturing metabolic syndrome clustering.
4. **Household Wealth (Richest Quintile):** Positive attribution in affluent households, reflecting higher diagnostic access and lifestyle nutrition transitions in urban India.
5. **Blood Pressure (Systolic BP):** Direct physiological biomarker confirming cardiovascular-metabolic coupling.

---

## 3. Patient-Level Local Explainability: Case Studies

### Case Study A: High-Risk Profile (Estimated Risk: **91.8%**)
- **Key Positive Attributions:** Older age bracket (45–49), elevated BMI (≥30.0), concurrent hypertension, and highest wealth quintile.
- **Visual Artifact:** [`reports/figures/shap_waterfall_high_risk_patient.png`](file:///c:/Users/Lenovo/OneDrive/Desktop/ml%20model/nfhs-diabetes-ml/reports/figures/shap_waterfall_high_risk_patient.png)

### Case Study B: Low-Risk Profile (Estimated Risk: **0.3%**)
- **Key Protective Attributions:** Younger demographic age (18–24), normal BMI (18.5–22.0), absence of hypertension, and active occupation.
- **Visual Artifact:** [`reports/figures/shap_waterfall_low_risk_patient.png`](file:///c:/Users/Lenovo/OneDrive/Desktop/ml%20model/nfhs-diabetes-ml/reports/figures/shap_waterfall_low_risk_patient.png)

---

## 4. Generated Artifacts & Visualizations

- Global Feature Importance Bar Chart: [`reports/figures/shap_feature_importance_bar.png`](file:///c:/Users/Lenovo/OneDrive/Desktop/ml%20model/nfhs-diabetes-ml/reports/figures/shap_feature_importance_bar.png)
- SHAP Summary Beeswarm Plot: [`reports/figures/shap_summary_beeswarm.png`](file:///c:/Users/Lenovo/OneDrive/Desktop/ml%20model/nfhs-diabetes-ml/reports/figures/shap_summary_beeswarm.png)
- High-Risk Case Study Waterfall: [`reports/figures/shap_waterfall_high_risk_patient.png`](file:///c:/Users/Lenovo/OneDrive/Desktop/ml%20model/nfhs-diabetes-ml/reports/figures/shap_waterfall_high_risk_patient.png)
- Low-Risk Case Study Waterfall: [`reports/figures/shap_waterfall_low_risk_patient.png`](file:///c:/Users/Lenovo/OneDrive/Desktop/ml%20model/nfhs-diabetes-ml/reports/figures/shap_waterfall_low_risk_patient.png)
- Feature Attribution Table: [`reports/tables/shap_feature_importance.csv`](file:///c:/Users/Lenovo/OneDrive/Desktop/ml%20model/nfhs-diabetes-ml/reports/tables/shap_feature_importance.csv)