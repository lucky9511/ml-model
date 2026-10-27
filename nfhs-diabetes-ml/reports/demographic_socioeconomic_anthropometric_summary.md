# Demographic, Socioeconomic, and Anthropometric Descriptive Summary

**Project:** `nfhs-diabetes-ml`  
**Dataset Analyzed:** `data/processed/analysis_dataset.parquet`  
**Sample Size:** 11,898 valid respondents (Women aged 15–49)  
**Target Variable:** `diabetes_status` (s728a: Self-reported diagnosed diabetes; 0 = Non-diabetic, 1 = Diabetic)  
**Verification Date:** 2026-10-22  
**Author:** Senior Machine Learning Engineer  

---

## 1. Executive Summary & Epistemic Stance

This report presents a structured exploratory descriptive analysis of the NFHS-5 Women's Individual Recode dataset across demographic, socioeconomic, and anthropometric domains. 

> [!IMPORTANT]
> **Descriptive Principles & Non-Causal Guidance:**  
> All statistics presented herein represent cross-sectional survey observations. **This descriptive pattern does not establish causality.** The observed differences reflect empirical distributions and correlations within the sample. Statements asserting causal mechanisms (such as "BMI causes diabetes" or "Wealth causes diabetes") are neither supported nor asserted by this cross-sectional exploratory evaluation.

---

## 2. Variables Analyzed

The exploratory assessment evaluated the following feature groups:

### Demographic Variables
- **Age (`age` / `v012`):** Continuous chronological age (15 to 49 years).
- **Age Groups (`age_group`):** Four demographic cohorts (`15-24`, `25-34`, `35-44`, `45-49`).
- **Place of Residence (`residence` / `v025`):** Binary classification (`Urban`, `Rural`).
- **State & Region (`state` / `region` / `v024`):** 36 States/UTs grouped into six administrative zones (`North`, `Central`, `East`, `Northeast`, `West`, `South`).
- **Marital Status (`marital_status` / `v502`):** Three categories (`Never married`, `Currently married`, `Formerly married`).

### Socioeconomic Variables
- **Highest Educational Attainment (`education` / `v106`):** Four ordinal levels (`No education`, `Primary`, `Secondary`, `Higher`).
- **Household Wealth Index (`wealth_group` / `v190`):** Five national quintiles (`Poorest`, `Poorer`, `Middle`, `Richer`, `Richest`).
- **Employment Status (`occupation` / `v714`):** Binary occupation proxy (`Not working`, `Currently working`).
- **Health Insurance Coverage (`health_insurance` / `v481`):** Binary coverage indicator (`Not covered`, `Covered`).
- **Household Survey Weight (`sample_weight` / `v005`):** Normalized DHS sampling weights.

### Anthropometric Variables
- **Standing Height (`height_cm` / `v438`):** Measured in centimeters.
- **Body Weight (`weight_kg` / `v447`):** Measured in kilograms.
- **Body Mass Index (`bmi` / `v445`):** Calculated continuous metric in $\text{kg/m}^2$.
- **BMI Categories (`bmi_category`):** WHO standard cutoffs (`Underweight [<18.5]`, `Normal [18.5-24.9]`, `Overweight [25.0-29.9]`, `Obese [>=30.0]`).
- **Vitals for Clinical Context:** Systolic blood pressure (`systolic_bp`) and diastolic blood pressure (`diastolic_bp`).

---

## 3. Main Descriptive Patterns

### Target Distribution & Severe Class Imbalance
The analyzed cohort comprises **11,898** respondents with a determinate diabetes diagnosis status:
- **Non-Diabetic ($Y = 0$):** 11,631 respondents (**97.76%**)
- **Diabetic ($Y = 1$):** 267 respondents (**2.24%**)
- **Imbalance Ratio:** **43.56 : 1**

### Demographic Patterns
1. **Age Gradient:**
   - The overall mean age of respondents is $32.16 \pm 10.04$ years (median: 32.0 years).
   - Observed diabetes prevalence displays a consistent upward gradient across chronological age groups:
     - `15-24` years: **0.54%** (18 / 3,304)
     - `25-34` years: **1.65%** (57 / 3,448)
     - `35-44` years: **3.18%** (109 / 3,432)
     - `45-49` years: **4.84%** (83 / 1,714)
   - **The target-positive group had a higher observed proportion of older respondents**, with the majority concentrated above 35 years of age.
2. **Urban–Rural Divergence:**
   - Rural respondents comprise **65.28%** of the sample ($n = 7,767$), while urban respondents comprise **34.72%** ($n = 4,131$).
   - **The target-positive group had a higher observed proportion of urban residents:** Observed prevalence in urban areas was **3.15%** (130 / 4,131), compared to **1.76%** (137 / 7,767) in rural areas.
3. **Marital Status:**
   - Currently married women constitute the largest cohort (71.84%, prevalence: **2.25%**). Never married women exhibited a prevalence of **2.47%**, and formerly married women had an observed prevalence of **1.45%**.
4. **Geographic Distribution:**
   - Regional observed prevalence varied moderately across administrative zones:
     - East: **2.51%** (32 / 1,273)
     - Northeast: **2.43%** (64 / 2,631)
     - South: **2.34%** (63 / 2,688)
     - Central: **2.19%** (23 / 1,051)
     - West: **2.04%** (27 / 1,324)
     - North: **1.98%** (58 / 2,931)

### Socioeconomic Patterns
1. **Wealth Index Quintiles:**
   - **The data show an association between higher household wealth quintiles and higher observed diabetes diagnosis prevalence:**
     - Poorest: **1.28%** (31 / 2,422)
     - Poorer: **1.61%** (38 / 2,362)
     - Middle: **2.22%** (53 / 2,390)
     - Richer: **2.50%** (60 / 2,396)
     - Richest: **3.65%** (85 / 2,328)
   - *Epistemic note:* This descriptive gradient does not establish that wealth causes diabetes. Rather, higher wealth is frequently associated with greater diagnostic healthcare access, urban sedentary employment, and dietary transitions.
2. **Educational Attainment:**
   - Women with no formal education exhibited an observed prevalence of **1.87%** (50 / 2,668), while those with primary education had **2.16%** (36 / 1,667), secondary education had **2.41%** (135 / 5,607), and higher education had **2.35%** (46 / 1,956).
3. **Occupation & Health Insurance:**
   - Non-working respondents had an observed prevalence of **2.36%** (188 / 7,967) compared to **1.95%** (74 / 3,797) among currently working respondents.
   - Insured respondents had an observed prevalence of **2.32%** (113 / 4,871), compared to **2.17%** (150 / 6,906) among uninsured respondents.

### Anthropometric Patterns
1. **Continuous Body Mass Index (BMI):**
   - The overall cleaned sample mean BMI is $26.66 \pm 3.62\text{ kg/m}^2$ (median: $26.65\text{ kg/m}^2$).
   - **The target-positive group had a higher observed mean BMI ($29.09 \pm 3.49\text{ kg/m}^2$, median: $29.48\text{ kg/m}^2$)** compared to the target-negative group ($26.61 \pm 3.60\text{ kg/m}^2$, median: $26.61\text{ kg/m}^2$).
2. **Categorical BMI Gradient:**
   - Underweight ($<18.5\text{ kg/m}^2$): **0.00%** (0 / 132)
   - Normal weight ($18.5\text{--}24.9\text{ kg/m}^2$): **0.87%** (31 / 3,571)
   - Overweight ($25.0\text{--}29.9\text{ kg/m}^2$): **2.09%** (119 / 5,692)
   - Obese ($\ge 30.0\text{ kg/m}^2$): **5.16%** (110 / 2,133)
   - **The data show a strong descriptive association between elevated BMI categories and higher observed diabetes prevalence.**

---

## 4. Quantitative Target-Group Differences Table

| Variable | Metric | Target Negative (Non-Diabetic, $n=11,631$) | Target Positive (Diabetic, $n=267$) | Absolute Difference |
| :--- | :---: | :---: | :---: | :---: |
| **Age (Years)** | Mean $\pm$ SD | $32.00 \pm 10.02$ | $39.09 \pm 8.09$ | $+7.09$ years |
| | Median [IQR] | $32.0$ [$23.0 - 41.0$] | $41.0$ [$34.0 - 46.0$] | $+9.0$ years |
| **Height (cm)** | Mean $\pm$ SD | $152.44 \pm 6.32$ | $152.67 \pm 6.64$ | $+0.23$ cm |
| | Median [IQR] | $152.4$ [$148.2 - 156.6$] | $152.0$ [$148.1 - 157.1$] | $-0.4$ cm |
| **Weight (kg)** | Mean $\pm$ SD | $61.90 \pm 9.89$ | $67.87 \pm 9.98$ | $+5.97$ kg |
| | Median [IQR] | $61.5$ [$54.9 - 68.4$] | $67.7$ [$60.5 - 75.2$] | $+6.2$ kg |
| **BMI ($\text{kg/m}^2$)** | Mean $\pm$ SD | $26.61 \pm 3.60$ | $29.09 \pm 3.49$ | $+2.48\text{ kg/m}^2$ |
| | Median [IQR] | $26.61$ [$24.17 - 29.06$] | $29.48$ [$26.44 - 31.46$] | $+2.87\text{ kg/m}^2$ |
| **Systolic BP (mmHg)** | Mean $\pm$ SD | $116.88 \pm 13.67$ | $123.44 \pm 13.31$ | $+6.56$ mmHg |
| | Median [IQR] | $117.0$ [$107.0 - 126.0$] | $123.0$ [$115.0 - 133.0$] | $+6.0$ mmHg |
| **Diastolic BP (mmHg)** | Mean $\pm$ SD | $74.87 \pm 8.91$ | $78.05 \pm 9.05$ | $+3.18$ mmHg |
| | Median [IQR] | $75.0$ [$69.0 - 81.0$] | $78.0$ [$72.0 - 84.0$] | $+3.0$ mmHg |
| **Urban Residence (%)** | Proportion | $34.39\%$ | $48.69\%$ | $+14.30\%$ |
| **Richest Wealth Quintile (%)**| Proportion | $19.28\%$ | $31.84\%$ | $+12.56\%$ |
| **Obese BMI $\ge 30$ (%)** | Proportion | $17.95\%$ | $42.31\%$ | $+24.36\%$ |

---

## 5. Missing-Data Observations

1. **Anthropometry Missingness (`v438`, `v447`, `v445`):**
   - 370 respondents (**3.11%**) were flagged with sentinel code `9998` (out of range, not measured, or refused).
   - In the target-positive group, 7 cases were missing BMI (2.62%), compared to 363 cases in the target-negative group (3.12%).
2. **Blood Pressure Missingness (`sb221`, `sb222`):**
   - 479 respondents (**4.03%**) had sentinel code `998` (not measured / refused).
   - Missingness was comparable across classes (5.24% in target-positive vs. 3.99% in target-negative).
3. **Socioeconomic Missingness:**
   - `occupation` had 134 missing records (**1.13%**).
   - `health_insurance` had 121 missing records (**1.02%**).
   - Missingness did not differ substantially across target groups.

---

## 6. Variables That Were Unavailable in the Dataset

The following requested anthropometric variables were **absent from the individual recode dataset**:
1. **Waist Circumference:** Not collected in the Women's Individual Recode questionnaire (IAIR7D) for this sample.
2. **Hip Circumference:** Not collected in this sample.
3. **Waist-to-Hip Ratio (WHR):** Could not be computed due to the absence of circumferences.
- *Remediation Note:* Central adiposity markers (WHR, waist circumference) are known markers of cardiometabolic risk in South Asian populations. However, in this specific DHS individual dataset, general adiposity is captured solely via BMI and body weight.

---

## 7. Limitations & Cautious Descriptive Guidance

1. **Cross-Sectional Confounding:** All survey observations are concurrent. While the target-positive group displays higher observed proportions of obesity, higher wealth, and urban residence, **this descriptive pattern does not establish causality**.
2. **Diagnostic Ascertainment Bias:** The target variable reflects *self-reported doctor diagnosis*. Affluent, urban, and older individuals have higher contact rates with healthcare professionals and greater screening frequency. Consequently, observed associations between higher wealth/urbanicity and diabetes diagnosis may partially reflect diagnostic access rather than biological incidence.
3. **Age Cohort Restriction:** The sample is restricted to women aged 15–49. The full epidemiological burden of Type 2 diabetes in India extends heavily into post-menopausal women aged 50 and above, meaning overall population prevalence is higher than observed in this reproductive-age cohort.
4. **Absence of Biological Glucose Measurements:** The dataset does not contain fasting plasma glucose or oral glucose tolerance test readings. Undiagnosed diabetes cannot be detected from this survey recode without external biomarker linkage.

---

## 8. Summary of Generated Visualizations

All 12 required figures were generated and verified in [`reports/figures/`](file:///c:/Users/Lenovo/OneDrive/Desktop/ml%20model/nfhs-diabetes-ml/reports/figures):

1. [`01_target_class_count_plot.png`](file:///c:/Users/Lenovo/OneDrive/Desktop/ml%20model/nfhs-diabetes-ml/reports/figures/01_target_class_count_plot.png) — Target class counts and percentages.
2. [`02_age_histogram.png`](file:///c:/Users/Lenovo/OneDrive/Desktop/ml%20model/nfhs-diabetes-ml/reports/figures/02_age_histogram.png) — Age distribution with mean and median markers.
3. [`03_age_distribution_by_target.png`](file:///c:/Users/Lenovo/OneDrive/Desktop/ml%20model/nfhs-diabetes-ml/reports/figures/03_age_distribution_by_target.png) — Stratified KDE density across diabetes status.
4. [`04_bmi_histogram.png`](file:///c:/Users/Lenovo/OneDrive/Desktop/ml%20model/nfhs-diabetes-ml/reports/figures/04_bmi_histogram.png) — Cleaned continuous BMI histogram with WHO cutoff reference.
5. [`05_bmi_distribution_by_target.png`](file:///c:/Users/Lenovo/OneDrive/Desktop/ml%20model/nfhs-diabetes-ml/reports/figures/05_bmi_distribution_by_target.png) — Stratified boxplot showing median and mean markers.
6. [`06_bmi_category_distribution.png`](file:///c:/Users/Lenovo/OneDrive/Desktop/ml%20model/nfhs-diabetes-ml/reports/figures/06_bmi_category_distribution.png) — Count and percentage breakdown across WHO BMI groups.
7. [`07_target_status_by_bmi_category.png`](file:///c:/Users/Lenovo/OneDrive/Desktop/ml%20model/nfhs-diabetes-ml/reports/figures/07_target_status_by_bmi_category.png) — Observed diabetes prevalence across BMI groups.
8. [`08_target_status_by_residence.png`](file:///c:/Users/Lenovo/OneDrive/Desktop/ml%20model/nfhs-diabetes-ml/reports/figures/08_target_status_by_residence.png) — Prevalence comparison between Urban and Rural strata.
9. [`09_target_status_by_education.png`](file:///c:/Users/Lenovo/OneDrive/Desktop/ml%20model/nfhs-diabetes-ml/reports/figures/09_target_status_by_education.png) — Prevalence across educational attainment tiers.
10. [`10_target_status_by_wealth_group.png`](file:///c:/Users/Lenovo/OneDrive/Desktop/ml%20model/nfhs-diabetes-ml/reports/figures/10_target_status_by_wealth_group.png) — Prevalence across household wealth quintiles.
11. [`11_target_status_by_age_group.png`](file:///c:/Users/Lenovo/OneDrive/Desktop/ml%20model/nfhs-diabetes-ml/reports/figures/11_target_status_by_age_group.png) — Prevalence across 10-year demographic age brackets.
12. [`12_target_status_by_region.png`](file:///c:/Users/Lenovo/OneDrive/Desktop/ml%20model/nfhs-diabetes-ml/reports/figures/12_target_status_by_region.png) — Prevalence across 6 Indian geographic zones.
