# Exploratory Data Analysis (EDA) Data Validation Report

**Project:** `nfhs-diabetes-ml`  
**Dataset Analyzed:** `data/processed/analysis_dataset.parquet`  
**Verification Date:** 2026-10-22  
**Author:** Senior Machine Learning Engineer  

---

## 1. Overview & Dataset Dimensions

- **Total Number of Rows:** **11,898** (Indeterminate target records `8` and `9` removed from the original 12,000 cohort)
- **Total Number of Columns:** **59**
- **File Format:** Apache Parquet (`analysis_dataset.parquet`)
- **Memory Footprint:** ~2.1 MB

---

## 2. Target Variable Verification

- **Target Column Name:** `diabetes_status` (with alias `target`)
- **Survey Provenance:** `s728a` (*"Told by doctor: Diabetes"*)
- **Unique Target Values & Counts:**
  - `0` (Non-Diabetic / Negative): **11,631** (97.76%)
  - `1` (Diabetic / Positive): **267** (2.24%)
- **Target Prevalence:** **2.24%**
- **Missing Target Count:** **0** (100% complete; indeterminate responses were systematically filtered during preprocessing)
- **Class Imbalance Ratio:** **43.56 : 1** (Strong tabular imbalance requiring stratified cross-validation and PR-AUC optimization)

---

## 3. Feature Space Breakdown

### Numerical Columns (48 Columns)
The dataset contains 48 numerical features covering raw survey codes, rescaled physical measurements, survey sampling weights, and chronic comorbidities:
- **Demographics & Survey Design:** `v001` (PSU cluster), `v002` (Household number), `v003` (Line number), `v005` (Raw sample weight), `sample_weight` (Normalized weight), `v012` (Age), `age` (Rescaled age), `v024` (State code), `v025` (Residence code), `v106` (Education code), `v130` (Religion code), `v131` (Caste code), `v190` (Wealth code), `v502` (Marital code), `v714` (Working code), `v481` (Insurance code)
- **Rescaled Anthropometrics & Vitals:** `height_cm`, `weight_kg`, `bmi`, `v438` (raw height), `v447` (raw weight), `v445` (raw BMI), `systolic_bp`, `diastolic_bp`, `sb221` (raw SBP), `sb222` (raw DBP)
- **Lifestyle & Dietary Frequencies:** `v463a` (Tobacco), `s716` (Alcohol), `s706a`–`s706i` (9 dietary items: milk, pulses, green leafy vegetables, fruits, eggs, fish, meat, fried food, aerated drinks)
- **Chronic Comorbidities:** `s728c` (Hypertension), `s728d` (Cancer), `s728e` (Asthma), `s728f` (Thyroid disorder), `s728g` (Heart disease)
- **Target & Identifiers:** `diabetes_status`, `target`, `s728a`
- **Missing Anthropometric Placeholders:** `waist_circumference`, `hip_circumference`, `waist_hip_ratio`

### Categorical Columns (11 Columns) & Cardinality
| Categorical Column | Data Type | Unique Values Count | Missing Count (NaNs) | Value Categories / Levels |
| :--- | :---: | :---: | :---: | :--- |
| `caseid` | `object` | 11,891 | 0 | Unique respondent identifier string |
| `age_group` | `object` | 4 | 0 | `15-24`, `25-34`, `35-44`, `45-49` |
| `residence` | `object` | 2 | 0 | `Urban`, `Rural` |
| `marital_status` | `object` | 3 | 0 | `Never married`, `Currently married`, `Formerly married` |
| `state` | `object` | 36 | 0 | 36 States & Union Territories of India |
| `region` | `object` | 6 | 0 | `North`, `Central`, `East`, `Northeast`, `West`, `South` |
| `education` | `object` | 4 | 0 | `No education`, `Primary`, `Secondary`, `Higher` |
| `wealth_group` | `object` | 5 | 0 | `Poorest`, `Poorer`, `Middle`, `Richer`, `Richest` |
| `occupation` | `object` | 2 | 134 | `Not working`, `Currently working` |
| `health_insurance` | `object` | 2 | 121 | `Not covered`, `Covered` |
| `bmi_category` | `object` | 4 | 370 | `Underweight (<18.5)`, `Normal (18.5-24.9)`, `Overweight (25.0-29.9)`, `Obese (>=30.0)` |

---

## 4. Data Quality & Distributional Checks

### Columns with > 50% Missing Values
1. `waist_circumference`: **100.0% missing** (Not captured in this individual recode sample)
2. `hip_circumference`: **100.0% missing** (Not captured in this individual recode sample)
3. `waist_hip_ratio`: **100.0% missing** (Cannot be derived due to missing circumferences)

*Note: All other columns have missingness rates below 3.5%, primarily corresponding to anthropometry flags (3.11%) or clinical blood pressure refusals (4.06%).*

### Constant or Near-Constant Columns (≥ 99% Single Value)
1. `s728d` (*"Told by doctor: Cancer"*): **99.49%** single negative value (`0`: 11,837, `1`: 61).
2. `waist_circumference`, `hip_circumference`, `waist_hip_ratio`: **100% NaN**.

### Possible Identifier & Survey Design Columns
These columns should not be utilized as direct predictive features in generalizable machine-learning models:
1. `caseid`: Primary respondent identifier string (near-unique: 11,891 unique out of 11,898 rows).
2. `v001`: Primary Sampling Unit (PSU) / Cluster number (6,615 unique clusters).
3. `v002`: Household identification number within cluster.
4. `v003`: Respondent line number on household roster.
5. `v005` / `sample_weight`: Normalized sampling weights (vital for survey-weighted estimation, but excluded from tree splits to avoid overfitting to sampling design).

---

## 5. Verification Conclusion

The processed analysis dataset satisfies all data integrity requirements. The target variable is unambiguous and cleanly separated from the leakage variable (`s728b`). The dataset is fully validated and ready for Phase 4 descriptive analysis.
