# Diabetes Variable Assessment Report: NFHS-5 India (2019–21)

**Project:** `nfhs-diabetes-ml`  
**Dataset Analyzed:** NFHS-5 Women's Individual Recode Sample (`NFHS5_sample_individual.dta`)  
**Sample Size:** 12,000 respondents | 38 variables  
**Author:** Senior Machine Learning Engineer  
**Date:** 2026-10-22  

---

## 1. Executive Summary

This assessment evaluates the candidate diabetes variables present in the NFHS-5 Individual Recode dataset for constructing an explainable tabular machine-learning classification system. We analyze variable semantics, value label mappings, survey routing/skip patterns, missingness codes, clinical validity, and critical **data leakage** risks.

The dataset contains two candidate diabetes-related variables:
1. `s728a`: **"Told by doctor: Diabetes"** (Self-reported medical diagnosis)
2. `s728b`: **"Currently taking medicine for diabetes (LEAKAGE)"** (Pharmacological treatment)

---

## 2. Candidate Diabetes Variables Profile

| Variable Code | Stata Variable Label | Data Type | Value Labels | Non-Null Count | Unique Values |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `s728a` | Told by doctor: Diabetes | `int64` | `0: No`<br>`1: Yes`<br>`8: Don't know`<br>`9: Missing` | 12,000 | 4 |
| `s728b` | Currently taking medicine for diabetes (LEAKAGE) | `int64` | `0: No`<br>`1: Yes`<br>`9: Missing` | 12,000 | 3 |

---

## 3. Detailed Frequency & Cross-Tabulation Analysis

### `s728a` Distribution (Diagnosis)
- **`0` (No):** 11,631 (96.925%)
- **`1` (Yes):** 267 (2.225%)
- **`8` (Don't know):** 54 (0.450%)
- **`9` (Missing):** 48 (0.400%)
- **Effective sample with known status (`0` or `1`):** 11,898 (99.15%)
- **Sample Prevalence:** **2.24%** (267 / 11,898)

### `s728b` Distribution (Medication)
- **`0` (No):** 11,684 (97.367%)
- **`1` (Yes):** 214 (1.783%)
- **`9` (Missing):** 102 (0.850%)

### Cross-Tabulation Matrix: `s728a` vs. `s728b`

| `s728a` (Diagnosed) \ `s728b` (Medication) | `0` (No Meds) | `1` (On Meds) | `9` (Missing / Skipped) | Total |
| :--- | :---: | :---: | :---: | :---: |
| **`0` (No)** | 11,622 | 9 | 0 | 11,631 |
| **`1` (Yes)** | 62 | 205 | 0 | 267 |
| **`8` (Don't know)** | 0 | 0 | 54 | 54 |
| **`9` (Missing)** | 0 | 0 | 48 | 48 |
| **Total** | **11,684** | **214** | **102** | **12,000** |

#### Clinical Insights from Cross-Tabulation:
1. **Medication Compliance / Treatment Rate:** Among respondents diagnosed with diabetes (`s728a == 1`), 76.8% (205 / 267) are on pharmacological treatment (`s728b == 1`), while 23.2% (62 / 267) manage through lifestyle/diet or lack access to medication.
2. **Off-Label / Misclassified Treatment:** 9 respondents who reported never being told by a doctor that they have diabetes (`s728a == 0`) reported taking diabetes medication (`s728b == 1`). This can stem from self-medication, pre-diabetes management, or survey misclassification.
3. **Survey Routing Logic:** All respondents who answered `8` ("Don't know") or `9` ("Missing") on `s728a` were systematically assigned `9` ("Missing") on `s728b` (exact sum: 54 + 48 = 102), confirming DHS skip-logic dependency.

---

## 4. Target Formulation & Mapping Strategies

### Recommended Primary Target: Binary Diagnosed Diabetes
```python
def map_primary_target(df):
    # 0 = No diabetes, 1 = Diagnosed diabetes, np.nan = indeterminate
    target = df['s728a'].map({0: 0, 1: 1, 8: np.nan, 9: np.nan})
    return target
```
- **Positive class ($Y = 1$):** 267 cases
- **Negative class ($Y = 0$):** 11,631 cases
- **Excluded indeterminate cases:** 102 cases (0.85%)
- **Target Prevalence:** 2.24%
- **Imbalance Ratio:** ~43.5 : 1 (Strong tabular class imbalance requiring PR-AUC scoring, balanced weights, or SMOTE/oversampling).

### Alternative Formulation: Composite Target (Diagnosed OR on Medication)
```python
def map_composite_target(df):
    target = np.where((df['s728a'] == 1) | (df['s728b'] == 1), 1,
             np.where((df['s728a'] == 0) & (df['s728b'] == 0), 0, np.nan))
    return target
```
- **Positive class ($Y = 1$):** 276 cases (2.32%)
- **Negative class ($Y = 0$):** 11,622 cases
- **Excluded:** 102 cases

---

## 5. Potential Data Leakage Risks

> [!WARNING]
> **CRITICAL DATA LEAKAGE:** `s728b` ("Currently taking medicine for diabetes") MUST NEVER be used as an input feature / predictor in any machine-learning model predicting diabetes status.

### Rationale:
- **Causal Inversion / Post-Treatment Variable:** Medication is initiated strictly *after* diagnosis. In a real-world clinical triage or early screening setting, patients being screened do not already take prescribed diabetes medication for an unknown condition.
- **Model Collapse:** If `s728b` is included in the feature set, tree models (XGBoost, LightGBM) will place near-100% split importance on this single feature, masking real epidemiological drivers (age, BMI, blood pressure, wealth, diet).
- **Rule for Pipeline:** `s728b` must be explicitly dropped from feature matrix $X$ prior to train/test splitting.

---

## 6. Suitability & Availability Assessment

| Criterion | Evaluation | Senior ML Engineer Notes |
| :--- | :---: | :--- |
| **Available for all respondents?** | **YES** | Fielded in the Women's Questionnaire across all 36 States/UTs for women aged 15–49. 99.15% complete response rate. |
| **Suitable as a supervised target?** | **YES** | Standard self-reported clinical diagnosis benchmark widely adopted in health economics and DHS epidemiology. |
| **Missing-code handling?** | **CLEAN** | Only 0.85% indeterminate records (`8` and `9`), which can be safely omitted from the training/evaluation split without introducing significant attrition bias. |
| **Measurement limitations?** | **ATTENTION** | Self-reported status captures *known* diabetes. It may underestimate total diabetes prevalence due to undiagnosed cases in lower socioeconomic or rural strata with limited diagnostic access. |

---

## 7. Requirement for Additional NFHS Files (Biomarker Analysis)

- **Current File (`NFHS5_sample_individual.dta`):** Contains self-reported diagnosis (`s728a`), anthropometry (height, weight, BMI), blood pressure measurements, and socio-demographic indicators.
- **Biomarker File Requirement:**
  - In NFHS-5, direct biological random blood glucose (RBG) testing was performed using portable glucometers on eligible respondents.
  - Biomarker values (e.g. glucose in mg/dL, fasting status, elevated blood glucose flag $\ge 140$ or $\ge 200$ mg/dL) are primarily recorded in the **Household Member Recode (`PR` file)** or the standalone **Biomarker Recode**.
  - **Verdict:** If the research objective is to detect **undiagnosed / biological hyperglycemia**, merging the DHS PR/Biomarker recode file via `v001` (PSU cluster), `v002` (Household number), and `v003` (Respondent line number) is recommended. For **diagnosed diabetes risk modeling and screening from survey data**, the current dataset is fully sufficient and ready for pipeline development.

---

## 8. Summary of Action Items for Pipeline Development

1. Set `s728a` as the primary classification target $Y$, with binary mapping `{0: 0, 1: 1}` and rows with `{8, 9}` filtered out.
2. Explicitly remove `s728b` from feature set $X$ to prevent direct target leakage.
3. Remove identifier and design columns from features (`caseid`, `v002`, `v003`).
4. Retain `v005` (sample weight) for survey-weighted risk evaluation and calibration.
5. Apply appropriate imbalanced learning techniques (PR-AUC, Stratified K-Fold, scale_pos_weight).
