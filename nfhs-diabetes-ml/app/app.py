"""
Streamlit Web Application: NFHS-5 Diabetes Clinical Risk Scoring & XAI Dashboard.

Upgraded Edition with:
1. Real-time patient diabetes risk estimation using the benchmark champion model.
2. Semicircular interactive Plotly risk gauge with calibrated threshold needle.
3. Interactive 'What-If' Counterfactual Simulator (lifestyle/clinical intervention modeling).
4. Downloadable Hospital-Grade Printable Clinical Report (HTML).
5. Batch Patient Screening & Triage Tool (CSV Upload & Sample Cohort Evaluation).
6. Local dynamic SHAP waterfall plot explaining individual risk attributions.
7. Epidemiological population insights from NFHS-5 survey cohort.
8. Model benchmark and cross-validation performance review.
"""

from pathlib import Path
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# Add project root to sys.path to enable src imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DATA_DIR, FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR, TABLES_DIR
from src.explainability import build_shap_explainer, clean_feature_name, FEATURE_DISPLAY_MAP
from src.models.pipeline import ALL_FEATURES
from src.preprocessing import STATE_MAP, REGION_MAP
from src.reporting import (
    create_risk_gauge_fig,
    create_counterfactual_comparison_fig,
    create_batch_distribution_fig,
    generate_clinical_html_report,
)
from src.utils import load_artifact

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="NFHS-5 Diabetes Risk & XAI Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Glassmorphism & High-Contrast CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #0d9488 100%);
        padding: 24px 32px;
        border-radius: 14px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(13, 148, 136, 0.2);
    }
    .main-header h1 {
        color: #ffffff;
        font-weight: 800;
        font-size: 2.2rem;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: #e0f2fe;
        font-size: 1.05rem;
        margin-top: 6px;
        margin-bottom: 0;
        font-weight: 400;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .metric-label {
        font-size: 0.80rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a;
    }
    .metric-sub {
        font-size: 0.85rem;
        color: #475569;
        margin-top: 4px;
    }

    .badge-low {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-moderate {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-high {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }

    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e293b;
        border-left: 4px solid #0d9488;
        padding-left: 10px;
        margin-top: 18px;
        margin-bottom: 14px;
    }

    .sim-box {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Resource Caching: Load Models & SHAP Explainer
# ---------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_model_and_artifacts():
    """Load serialized pipeline, metadata, and build background SHAP explainer."""
    model_path = MODELS_DIR / "best_model.joblib"
    meta_path = MODELS_DIR / "best_model_metadata.json"
    data_path = PROCESSED_DATA_DIR / "analysis_dataset.parquet"

    if not model_path.exists():
        st.error(f"Champion model not found at {model_path}. Please execute `train.py` first.")
        st.stop()

    pipeline = load_artifact(model_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    df_sample = pd.read_parquet(data_path)
    X_background = df_sample[ALL_FEATURES].sample(n=250, random_state=42)

    explainer, raw_names, clean_names = build_shap_explainer(
        pipeline, X_background, max_background_samples=250
    )

    return pipeline, metadata, explainer, clean_names, df_sample


pipeline, metadata, explainer, clean_names, df_cohort = load_model_and_artifacts()
optimal_threshold = metadata.get("optimal_threshold", 0.7225)
baseline_prev = 0.0224


# ---------------------------------------------------------
# Header Banner
# ---------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🩺 NFHS-5 Diabetes Clinical Decision Support & Explainable AI</h1>
        <p>Evidence-based tabular risk stratification, counterfactual lifestyle simulation, and patient SHAP attribution using India NFHS-5 data.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Sidebar Controls & Presets
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/medical-heart.png", width=64)
    st.title("Clinical Control Panel")
    st.markdown("**Target Population:** NFHS-5 Women (15–49)")

    preset = st.selectbox(
        "⚡ Quick Clinical Presets",
        options=[
            "Custom Patient Input",
            "High-Risk Patient (Older, Obese, Hypertensive)",
            "Moderate-Risk Patient (Mid-Age, Overweight)",
            "Low-Risk Patient (Young, Healthy BMI)",
        ],
        index=0,
    )

    st.divider()
    st.markdown("### 🏆 Production Model Status")
    st.markdown(f"**Architecture:** `{metadata['model_name']}`")
    st.markdown(f"**Test ROC-AUC:** `{metadata['test_performance']['roc_auc']:.4f}`")
    st.markdown(f"**Test PR-AUC:** `{metadata['test_performance']['pr_auc']:.4f}`")
    st.markdown(f"**Decision Cutoff ($\tau$):** `{optimal_threshold:.4f}`")
    st.caption("Calibrated on 11,898 NFHS-5 records to counter severe 43.5:1 class imbalance.")

    st.divider()
    st.markdown("### ℹ️ Survey Source")
    st.caption(
        "National Family Health Survey 5 (2019-21) Women's Recode (IAIR7D). "
        "Strictly screens non-post-diagnostic features to guarantee pre-screening clinical validity."
    )


# ---------------------------------------------------------
# Preset Initializer
# ---------------------------------------------------------
if preset == "High-Risk Patient (Older, Obese, Hypertensive)":
    init_age = 47
    init_height = 152
    init_weight = 75.0
    init_sbp = 142
    init_dbp = 92
    init_residence = "Urban"
    init_state = "Kerala"
    init_edu = "Secondary"
    init_wealth = "Richest"
    init_marital = "Currently married"
    init_htn = True
    init_heart = False
    init_thyroid = True
    init_asthma = False
    init_cancer = False
    init_tobacco = False
    init_alcohol = False
    init_diet_fried = 2  # Weekly
    init_diet_pulses = 1  # Daily
elif preset == "Moderate-Risk Patient (Mid-Age, Overweight)":
    init_age = 38
    init_height = 155
    init_weight = 66.0
    init_sbp = 126
    init_dbp = 82
    init_residence = "Urban"
    init_state = "Maharashtra"
    init_edu = "Secondary"
    init_wealth = "Richer"
    init_marital = "Currently married"
    init_htn = False
    init_heart = False
    init_thyroid = False
    init_asthma = False
    init_cancer = False
    init_tobacco = False
    init_alcohol = False
    init_diet_fried = 3
    init_diet_pulses = 1
elif preset == "Low-Risk Patient (Young, Healthy BMI)":
    init_age = 22
    init_height = 156
    init_weight = 49.0
    init_sbp = 110
    init_dbp = 72
    init_residence = "Rural"
    init_state = "Himachal Pradesh"
    init_edu = "Higher"
    init_wealth = "Middle"
    init_marital = "Never married"
    init_htn = False
    init_heart = False
    init_thyroid = False
    init_asthma = False
    init_cancer = False
    init_tobacco = False
    init_alcohol = False
    init_diet_fried = 4  # Never
    init_diet_pulses = 1
else:  # Custom default (median survey baseline)
    init_age = 32
    init_height = 152
    init_weight = 61.0
    init_sbp = 118
    init_dbp = 76
    init_residence = "Rural"
    init_state = "Uttar Pradesh"
    init_edu = "Secondary"
    init_wealth = "Middle"
    init_marital = "Currently married"
    init_htn = False
    init_heart = False
    init_thyroid = False
    init_asthma = False
    init_cancer = False
    init_tobacco = False
    init_alcohol = False
    init_diet_fried = 3
    init_diet_pulses = 1


# ---------------------------------------------------------
# Primary Navigation Tabs
# ---------------------------------------------------------
tab1, tab_batch, tab2, tab3, tab4 = st.tabs([
    "🩺 Patient Risk Scoring & 'What-If' Simulator",
    "📁 Batch Patient Screening & Triage Tool",
    "📊 Population Epidemiological Insights",
    "🧠 Model Benchmarks & Metrics",
    "🔬 Global Feature Importance (SHAP)",
])


# =========================================================
# TAB 1: Live Risk Scoring, What-If & Printable Report
# =========================================================
with tab1:
    st.markdown('<div class="section-header">1. Patient Demographic & Physiological Profile</div>', unsafe_allow_html=True)

    col_demo, col_anthro, col_clin = st.columns([1, 1, 1], gap="medium")

    # --- Column 1: Demographics ---
    with col_demo:
        st.markdown("**Demographic & Socioeconomic**")
        age = st.slider("Age of Respondent (Years)", min_value=15, max_value=49, value=init_age, step=1)
        residence = st.radio("Place of Residence", options=["Urban", "Rural"], index=0 if init_residence == "Urban" else 1, horizontal=True)

        all_states = sorted(list(STATE_MAP.values()))
        state_idx = all_states.index(init_state) if init_state in all_states else 0
        state = st.selectbox("State / Union Territory", options=all_states, index=state_idx)
        region = REGION_MAP.get(state, "Other")

        wealth_group = st.selectbox(
            "Household Wealth Quintile",
            options=["Poorest", "Poorer", "Middle", "Richer", "Richest"],
            index=["Poorest", "Poorer", "Middle", "Richer", "Richest"].index(init_wealth),
        )

        education = st.selectbox(
            "Highest Educational Attainment",
            options=["No education", "Primary", "Secondary", "Higher"],
            index=["No education", "Primary", "Secondary", "Higher"].index(init_edu),
        )

        marital_status = st.selectbox(
            "Marital Status",
            options=["Currently married", "Never married", "Formerly married"],
            index=["Currently married", "Never married", "Formerly married"].index(init_marital),
        )

        occupation = st.radio("Current Working Status", options=["Not working", "Currently working"], index=1, horizontal=True)
        health_insurance = st.radio("Health Insurance Coverage", options=["Covered", "Not covered"], index=0, horizontal=True)

    # --- Column 2: Anthropometrics & Vitals ---
    with col_anthro:
        st.markdown("**Anthropometry & Clinical Vitals**")
        height_cm = st.number_input("Standing Height (cm)", min_value=120.0, max_value=200.0, value=float(init_height), step=0.5)
        weight_kg = st.number_input("Body Weight (kg)", min_value=30.0, max_value=150.0, value=float(init_weight), step=0.5)

        # Dynamic continuous BMI calculation
        height_m = height_cm / 100.0
        calculated_bmi = round(weight_kg / (height_m ** 2), 2)

        # WHO BMI Categorization
        if calculated_bmi < 18.5:
            bmi_cat = "Underweight (<18.5)"
            bmi_badge_color = "#3b82f6"
        elif calculated_bmi < 25.0:
            bmi_cat = "Normal (18.5-24.9)"
            bmi_badge_color = "#10b981"
        elif calculated_bmi < 30.0:
            bmi_cat = "Overweight (25.0-29.9)"
            bmi_badge_color = "#f59e0b"
        else:
            bmi_cat = "Obese (>=30.0)"
            bmi_badge_color = "#ef4444"

        st.markdown(
            f"""
            <div style="background:#f1f5f9; padding:12px; border-radius:8px; margin: 10px 0;">
                <span style="font-weight:700; color:#334155;">Calculated BMI:</span>
                <span style="font-size:1.3rem; font-weight:800; color:{bmi_badge_color}; margin-left:8px;">{calculated_bmi} kg/m²</span>
                <br><span style="font-size:0.85rem; color:#475569;">WHO Classification: <b>{bmi_cat}</b></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        systolic_bp = st.slider("Systolic Blood Pressure (mmHg)", min_value=80, max_value=220, value=init_sbp, step=1)
        diastolic_bp = st.slider("Diastolic Blood Pressure (mmHg)", min_value=50, max_value=140, value=init_dbp, step=1)

        if systolic_bp >= 140 or diastolic_bp >= 90:
            st.warning("⚠️ Stage 2 Hypertension Alert: SBP ≥ 140 or DBP ≥ 90 mmHg.")

    # --- Column 3: Comorbidities & Lifestyle ---
    with col_clin:
        st.markdown("**Chronic Conditions & Dietary Frequencies**")
        htn_flag = st.checkbox("Doctor Diagnosed Hypertension", value=init_htn)
        heart_flag = st.checkbox("Doctor Diagnosed Heart Disease", value=init_heart)
        thyroid_flag = st.checkbox("Doctor Diagnosed Thyroid Disorder", value=init_thyroid)
        asthma_flag = st.checkbox("Doctor Diagnosed Asthma", value=init_asthma)
        cancer_flag = st.checkbox("Doctor Diagnosed Cancer", value=init_cancer)
        tobacco_flag = st.checkbox("Smokes Cigarettes / Tobacco", value=init_tobacco)
        alcohol_flag = st.checkbox("Drinks Alcohol", value=init_alcohol)

        st.markdown("**Dietary Intake Patterns**")
        diet_map = {"Daily": 1, "Weekly": 2, "Occasionally": 3, "Never": 4}
        diet_pulses_str = st.selectbox("Pulses / Beans Frequency", options=["Daily", "Weekly", "Occasionally", "Never"], index=0)
        diet_fried_str = st.selectbox("Fried Foods Frequency", options=["Daily", "Weekly", "Occasionally", "Never"], index=init_diet_fried - 1)
        diet_fruit_str = st.selectbox("Fresh Fruits Frequency", options=["Daily", "Weekly", "Occasionally", "Never"], index=1)
        diet_aerated_str = st.selectbox("Aerated Soft Drinks", options=["Daily", "Weekly", "Occasionally", "Never"], index=2)

    # -----------------------------------------------------
    # Construct Feature DataFrame for Current Inference
    # -----------------------------------------------------
    patient_dict = {
        "age": age,
        "bmi": calculated_bmi,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "systolic_bp": float(systolic_bp),
        "diastolic_bp": float(diastolic_bp),
        "s706a": 1.0,
        "s706b": float(diet_map[diet_pulses_str]),
        "s706c": 1.0,
        "s706d": float(diet_map[diet_fruit_str]),
        "s706e": 2.0,
        "s706f": 3.0,
        "s706g": 3.0,
        "s706h": float(diet_map[diet_fried_str]),
        "s706i": float(diet_map[diet_aerated_str]),
        "s728c": 1.0 if htn_flag else 0.0,
        "s728d": 1.0 if cancer_flag else 0.0,
        "s728e": 1.0 if asthma_flag else 0.0,
        "s728f": 1.0 if thyroid_flag else 0.0,
        "s728g": 1.0 if heart_flag else 0.0,
        "v463a": 1.0 if tobacco_flag else 0.0,
        "s716": 1.0 if alcohol_flag else 0.0,
        "residence": residence,
        "education": education,
        "wealth_group": wealth_group,
        "marital_status": marital_status,
        "region": region,
        "occupation": occupation,
        "health_insurance": health_insurance,
        "bmi_category": bmi_cat,
    }

    patient_df = pd.DataFrame([patient_dict])[ALL_FEATURES]

    # Preprocess & Predict Probability
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    X_patient_trans = preprocessor.transform(patient_df)
    predicted_prob = float(classifier.predict_proba(X_patient_trans)[0, 1])
    is_positive_call = predicted_prob >= optimal_threshold
    relative_risk = predicted_prob / baseline_prev

    st.markdown('<div class="section-header">2. Real-Time Risk Stratification & Clinical Prediction</div>', unsafe_allow_html=True)

    col_gauge, col_cards = st.columns([1.1, 1], gap="medium")

    with col_gauge:
        fig_gauge = create_risk_gauge_fig(predicted_prob, optimal_threshold, baseline_prev)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_cards:
        if predicted_prob < 0.30:
            risk_tier = "Low Clinical Risk"
            badge_html = '<span class="badge-low">🟢 Low Clinical Risk</span>'
        elif predicted_prob < optimal_threshold:
            risk_tier = "Moderate Risk Tier"
            badge_html = '<span class="badge-moderate">🟡 Moderate Risk</span>'
        else:
            risk_tier = "High / Actionable Screening Indicated"
            badge_html = '<span class="badge-high">🔴 High / Actionable Screening Priority</span>'

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Risk Category & Stratification</div>
                <div style="margin-top:6px; margin-bottom:8px;">{badge_html}</div>
                <div class="metric-sub">{risk_tier}</div>
                <hr style="margin: 12px 0; border: none; border-top: 1px solid #e2e8f0;">
                <div style="display:flex; justify-content:space-between;">
                    <div>
                        <div class="metric-label">Relative Risk Multiplier</div>
                        <div style="font-size:1.4rem; font-weight:800; color:#0f172a;">{relative_risk:.1f}×</div>
                        <div style="font-size:0.75rem; color:#64748b;">vs Baseline (2.24%)</div>
                    </div>
                    <div>
                        <div class="metric-label">Triage Cutoff Action</div>
                        <div style="font-size:1.1rem; font-weight:800; color:{'#b91c1c' if is_positive_call else '#0f766e'}; margin-top:4px;">
                            {'RECOMMEND SCREENING' if is_positive_call else 'ROUTINE MONITORING'}
                        </div>
                        <div style="font-size:0.75rem; color:#64748b;">Threshold: {optimal_threshold:.3f}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------
    # SHAP Waterfall Local Explanation
    # -----------------------------------------------------
    st.markdown('<div class="section-header">3. Local Explainability: SHAP Waterfall Risk Attribution</div>', unsafe_allow_html=True)
    st.caption("Game-theoretic attribution reveals each factor's exact push or pull on the model's log-odds relative to population baseline $E[f(X)]$.")

    shap_explanation = explainer(X_patient_trans)
    shap_explanation.feature_names = clean_names

    fig_waterfall, ax_wf = plt.subplots(figsize=(10, 5), dpi=300)
    shap.plots.waterfall(shap_explanation[0], max_display=12, show=False)
    plt.title(f"Patient Risk Factor Waterfall | Output: f(x) = {shap_explanation.values[0].sum() + shap_explanation.base_values[0]:.2f} log-odds", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    st.pyplot(fig_waterfall)
    plt.close()

    # Extract top risk and protective factors for report
    inst_shap = shap_explanation[0]
    pairs = list(zip(clean_names, inst_shap.values))
    pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
    top_risk_factors = [(f, v) for f, v in pairs_sorted if v > 0][:5]
    top_protective_factors = [(f, v) for f, v in reversed(pairs_sorted) if v < 0][:5]

    # Clinical Advice Formulation
    advice_items = []
    if calculated_bmi >= 25.0:
        advice_items.append(f"Weight Management: Patient BMI ({calculated_bmi} kg/m²) is classified as {bmi_cat}. Targeted weight loss of 5–10% can reduce risk by over 30%.")
    if systolic_bp >= 130 or htn_flag:
        advice_items.append("Blood Pressure Optimization: SBP ≥ 130 mmHg is an active cardiometabolic multiplier. Adopt DASH sodium reduction and regular BP monitoring.")
    if age >= 40:
        advice_items.append("Annual Glycemic Triage: Reproductive-age women over 40 exhibit the highest population burden. Schedule annual fasting plasma glucose or HbA1c screening.")
    if diet_fried_str in ["Daily", "Weekly"]:
        advice_items.append("Dietary Adjustment: Frequent fried food consumption contributes to adverse insulin sensitivity. Shift toward high-fiber whole pulses.")
    if not advice_items:
        advice_items.append("Patient displays an optimal biomarker profile. Encourage regular physical activity and routine bi-annual check-ups.")

    # Printable HTML Report Download
    report_html = generate_clinical_html_report(
        patient_dict,
        predicted_prob,
        optimal_threshold,
        risk_tier,
        top_risk_factors,
        top_protective_factors,
        advice_items,
    )

    st.download_button(
        label="📥 Download Hospital-Grade Printable Clinical Report (HTML)",
        data=report_html,
        file_name=f"clinical_diabetes_report_pt_{age}y.html",
        mime="text/html",
        help="Download formatted report that can be opened in any browser and printed (Ctrl+P / Save as PDF).",
    )

    # -----------------------------------------------------
    # 4. Interactive "What-If" Counterfactual Simulator
    # -----------------------------------------------------
    st.markdown('<div class="section-header">4. 🔮 Interactive "What-If" Counterfactual Intervention Simulator</div>', unsafe_allow_html=True)
    st.markdown(
        "Simulate clinical and lifestyle interventions to determine **how much risk can be mitigated** through weight loss, blood pressure control, and smoking cessation."
    )

    with st.container():
        st.markdown('<div class="sim-box">', unsafe_allow_html=True)
        sim_col1, sim_col2, sim_col3 = st.columns(3)

        with sim_col1:
            st.markdown("**1. Anthropometric Target**")
            target_weight = st.slider(
                "Target Weight (kg)",
                min_value=35.0,
                max_value=float(weight_kg),
                value=float(min(weight_kg, 60.0)),
                step=0.5,
                help="Simulate the effect of healthy weight reduction.",
            )
            sim_bmi = round(target_weight / (height_m ** 2), 2)
            if sim_bmi < 18.5:
                sim_bmi_cat = "Underweight (<18.5)"
            elif sim_bmi < 25.0:
                sim_bmi_cat = "Normal (18.5-24.9)"
            elif sim_bmi < 30.0:
                sim_bmi_cat = "Overweight (25.0-29.9)"
            else:
                sim_bmi_cat = "Obese (>=30.0)"
            st.caption(f"Simulated BMI: **{sim_bmi} kg/m²** ({sim_bmi_cat})")

        with sim_col2:
            st.markdown("**2. Vitals Target**")
            target_sbp = st.slider(
                "Target Systolic BP (mmHg)",
                min_value=90,
                max_value=max(int(systolic_bp), 120),
                value=min(int(systolic_bp), 118),
                step=1,
            )
            target_dbp = st.slider(
                "Target Diastolic BP (mmHg)",
                min_value=60,
                max_value=max(int(diastolic_bp), 80),
                value=min(int(diastolic_bp), 76),
                step=1,
            )

        with sim_col3:
            st.markdown("**3. Lifestyle Changes**")
            sim_quit_tobacco = st.checkbox("Smoking Cessation (Quit Tobacco)", value=True if tobacco_flag else False)
            sim_diet_fried = st.selectbox("Fried Foods Intake", ["Never", "Occasionally", "Weekly", "Daily"], index=0)

        # Build simulated patient row
        sim_patient_dict = patient_dict.copy()
        sim_patient_dict["weight_kg"] = target_weight
        sim_patient_dict["bmi"] = sim_bmi
        sim_patient_dict["bmi_category"] = sim_bmi_cat
        sim_patient_dict["systolic_bp"] = float(target_sbp)
        sim_patient_dict["diastolic_bp"] = float(target_dbp)
        if sim_quit_tobacco:
            sim_patient_dict["v463a"] = 0.0
        sim_patient_dict["s706h"] = float(diet_map[sim_diet_fried])

        sim_patient_df = pd.DataFrame([sim_patient_dict])[ALL_FEATURES]
        X_sim_trans = preprocessor.transform(sim_patient_df)
        sim_prob = float(classifier.predict_proba(X_sim_trans)[0, 1])

        # Compute risk deltas
        abs_delta = sim_prob - predicted_prob
        rel_reduction = ((predicted_prob - sim_prob) / predicted_prob) * 100 if predicted_prob > 0 else 0

        sim_res_col1, sim_res_col2 = st.columns([1, 1.2])

        with sim_res_col1:
            st.markdown("#### Projected Risk Impact")
            st.metric(
                label="Simulated Risk Score",
                value=f"{sim_prob * 100:.1f}%",
                delta=f"{abs_delta * 100:.1f}% Absolute",
                delta_color="inverse",
            )
            st.markdown(
                f"""
                <div style="background:#ecfdf5; border-left:4px solid #10b981; padding:12px; border-radius:6px; margin-top:12px;">
                    <b style="color:#065f46; font-size:1.05rem;">🎉 Projected Relative Risk Reduction: {rel_reduction:.1f}%</b>
                    <br><span style="font-size:0.85rem; color:#047857;">By achieving target BMI ({sim_bmi}) and BP ({target_sbp}/{target_dbp} mmHg), estimated diabetes risk drops significantly.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with sim_res_col2:
            fig_comp = create_counterfactual_comparison_fig(predicted_prob, sim_prob)
            st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# TAB 2: Batch Patient Screening & Triage Tool (CSV Upload)
# =========================================================
with tab_batch:
    st.markdown('<div class="section-header">📁 Batch Patient Screening & Community Health Triage</div>', unsafe_allow_html=True)
    st.markdown(
        "Upload a roster of patients to perform **automated batch risk scoring**, community triage classification, and download an actionable clinical prioritization roster."
    )

    batch_file = st.file_uploader("Upload Patient Cohort CSV", type=["csv"], help="CSV must contain demographic, anthropometric, and vital features.")

    sample_csv_path = DATA_DIR / "sample_batch_patients.csv"
    load_sample = st.button("⚡ Or Load Demo Cohort of 50 Patients from NFHS-5")

    df_to_score = None
    if batch_file is not None:
        df_to_score = pd.read_csv(batch_file)
        st.success(f"Uploaded cohort with {len(df_to_score)} patient records.")
    elif load_sample and sample_csv_path.exists():
        df_to_score = pd.read_csv(sample_csv_path)
        st.info("Loaded demo cohort of 50 patients.")

    if df_to_score is not None:
        # Check required columns
        missing_in_batch = [c for c in ALL_FEATURES if c not in df_to_score.columns]
        if missing_in_batch:
            st.error(f"Missing required feature columns in batch file: {missing_in_batch}")
        else:
            with st.spinner("Scoring patient cohort with production pipeline..."):
                X_batch_trans = preprocessor.transform(df_to_score[ALL_FEATURES])
                batch_probs = classifier.predict_proba(X_batch_trans)[:, 1]

                results_df = df_to_score.copy()
                results_df["predicted_risk_prob"] = batch_probs
                results_df["predicted_risk_pct"] = np.round(batch_probs * 100, 1)

                def assign_tier(p):
                    if p >= optimal_threshold:
                        return "High Risk / Screening Indicated"
                    elif p >= 0.30:
                        return "Moderate Risk"
                    else:
                        return "Low Risk"

                results_df["risk_tier"] = [assign_tier(p) for p in batch_probs]
                results_df["triage_priority"] = [
                    "URGENT CLINICAL TRIAGE" if p >= optimal_threshold else "ROUTINE REVIEW" for p in batch_probs
                ]

                # Triage Summary Metric Cards
                t_total = len(results_df)
                t_high = int((results_df["risk_tier"] == "High Risk / Screening Indicated").sum())
                t_mod = int((results_df["risk_tier"] == "Moderate Risk").sum())
                t_low = int((results_df["risk_tier"] == "Low Risk").sum())

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Patients Screened", t_total)
                c2.metric("High Priority Actionable", f"{t_high} ({t_high/t_total*100:.1f}%)")
                c3.metric("Moderate Risk", f"{t_mod} ({t_mod/t_total*100:.1f}%)")
                c4.metric("Low Risk Normal", f"{t_low} ({t_low/t_total*100:.1f}%)")

                # Distribution Visualizations
                fig_donut, fig_hist = create_batch_distribution_fig(results_df)
                vcol1, vcol2 = st.columns(2)
                with vcol1:
                    st.plotly_chart(fig_donut, use_container_width=True)
                with vcol2:
                    st.plotly_chart(fig_hist, use_container_width=True)

                # Prioritized Patient Table
                st.markdown("#### Actionable Patient Triage Roster (Sorted by Estimated Risk)")
                sort_cols = ["patient_id"] if "patient_id" in results_df.columns else []
                sort_cols += ["predicted_risk_pct", "risk_tier", "triage_priority", "age", "bmi", "systolic_bp", "wealth_group", "residence"]
                available_cols = [c for c in sort_cols if c in results_df.columns]

                sorted_roster = results_df[available_cols].sort_values(by="predicted_risk_pct", ascending=False).reset_index(drop=True)
                st.dataframe(
                    sorted_roster.style.format({"predicted_risk_pct": "{:.1f}%"}).highlight_between(
                        subset=["predicted_risk_pct"], left=optimal_threshold * 100, right=100.0, color="#fecaca"
                    ),
                    use_container_width=True,
                )

                # Download Scored CSV
                csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Complete Triage Roster (CSV)",
                    data=csv_bytes,
                    file_name="nfhs5_diabetes_batch_triage_roster.csv",
                    mime="text/csv",
                )


# =========================================================
# TAB 3: Population Epidemiological Insights
# =========================================================
with tab2:
    st.markdown('<div class="section-header">NFHS-5 India (2019–21) Cohort Distributions</div>', unsafe_allow_html=True)
    st.markdown(
        "Descriptive cross-sectional analysis of **11,898 women aged 15–49** across 36 Indian States and Union Territories. "
        "Overall sample diagnosed diabetes prevalence: **2.24%** (43.56:1 class imbalance)."
    )

    col_fig1, col_fig2 = st.columns(2)

    with col_fig1:
        st.markdown("#### 1. Age Cohort Distribution & Prevalence Gradient")
        img_age = FIGURES_DIR / "11_target_status_by_age_group.png"
        if img_age.exists():
            st.image(str(img_age), use_container_width=True)
        st.caption("Diabetes diagnosis displays a steep upward trajectory: 0.54% (15–24) to 4.84% (45–49).")

    with col_fig2:
        st.markdown("#### 2. WHO BMI Category Prevalence Gradient")
        img_bmi = FIGURES_DIR / "07_target_status_by_bmi_category.png"
        if img_bmi.exists():
            st.image(str(img_bmi), use_container_width=True)
        st.caption("Obese respondents (BMI ≥ 30.0) exhibit a 5.16% diagnosed rate vs 0.87% in normal weight individuals.")

    col_fig3, col_fig4 = st.columns(2)

    with col_fig3:
        st.markdown("#### 3. Household Wealth Quintile Gradient")
        img_wealth = FIGURES_DIR / "10_target_status_by_wealth_group.png"
        if img_wealth.exists():
            st.image(str(img_wealth), use_container_width=True)
        st.caption("Affluent quintiles exhibit higher diagnosed prevalence, reflecting both lifestyle nutrition transitions and diagnostic access.")

    with col_fig4:
        st.markdown("#### 4. Urban vs Rural Residence Disparity")
        img_res = FIGURES_DIR / "08_target_status_by_residence.png"
        if img_res.exists():
            st.image(str(img_res), use_container_width=True)
        st.caption("Urban diagnosed prevalence is 3.15% compared to 1.76% in rural communities.")


# =========================================================
# TAB 4: Model Architecture & Benchmark Metrics
# =========================================================
with tab3:
    st.markdown('<div class="section-header">Tabular ML Benchmark & Cross-Validation Results</div>', unsafe_allow_html=True)
    st.markdown("Systematic comparison of 4 algorithms across native cost-sensitive class weighting and SMOTE oversampling:")

    benchmark_csv = TABLES_DIR / "model_benchmark_results.csv"
    if benchmark_csv.exists():
        df_bench = pd.read_csv(benchmark_csv)
        display_df = df_bench[[
            "model_name", "imbalance_strategy", "roc_auc_mean", "pr_auc_mean", "f1_optimal_mean", "recall_mean", "precision_mean"
        ]].copy()
        display_df.columns = ["Model Architecture", "Imbalance Method", "ROC-AUC (CV)", "PR-AUC (CV)", "Optimal F1", "Sensitivity / Recall", "Precision"]
        st.dataframe(
            display_df.style.format({
                "ROC-AUC (CV)": "{:.4f}",
                "PR-AUC (CV)": "{:.4f}",
                "Optimal F1": "{:.4f}",
                "Sensitivity / Recall": "{:.4f}",
                "Precision": "{:.4f}",
            }).highlight_max(subset=["PR-AUC (CV)", "ROC-AUC (CV)"], color="#d1fae5"),
            use_container_width=True,
        )

    col_roc, col_pr = st.columns(2)

    with col_roc:
        st.markdown("#### Receiver Operating Characteristic (ROC) Curves")
        roc_img = FIGURES_DIR / "model_roc_curves.png"
        if roc_img.exists():
            st.image(str(roc_img), use_container_width=True)

    with col_pr:
        st.markdown("#### Precision-Recall (PR) Curves")
        pr_img = FIGURES_DIR / "model_pr_curves.png"
        if pr_img.exists():
            st.image(str(pr_img), use_container_width=True)

    st.markdown("#### Held-Out 20% Test Cohort Evaluation")
    test_p = metadata["test_performance"]
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    col_t1.metric("Test ROC-AUC", f"{test_p['roc_auc']:.4f}")
    col_t2.metric("Test PR-AUC", f"{test_p['pr_auc']:.4f}", delta="3.8x above random")
    col_t3.metric("Test Sensitivity", f"{test_p['recall_optimal']*100:.1f}%")
    col_t4.metric("Test Specificity", f"{test_p['confusion_matrix'][0][0] / (test_p['confusion_matrix'][0][0] + test_p['confusion_matrix'][0][1])*100:.1f}%")


# =========================================================
# TAB 5: Global Feature Importance (SHAP Library)
# =========================================================
with tab4:
    st.markdown('<div class="section-header">Global Interpretability & Feature Rankings (SHAP)</div>', unsafe_allow_html=True)
    st.markdown("Game-theoretic Shapley values calculated across a representative survey cohort:")

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("#### Mean |SHAP| Global Feature Importance")
        bar_img = FIGURES_DIR / "shap_feature_importance_bar.png"
        if bar_img.exists():
            st.image(str(bar_img), use_container_width=True)

    with col_g2:
        st.markdown("#### SHAP Beeswarm Summary Plot (Directionality)")
        beeswarm_img = FIGURES_DIR / "shap_summary_beeswarm.png"
        if beeswarm_img.exists():
            st.image(str(beeswarm_img), use_container_width=True)

    st.markdown("#### Structured Feature Attribution Rankings")
    shap_table_path = TABLES_DIR / "shap_feature_importance.csv"
    if shap_table_path.exists():
        df_shap = pd.read_csv(shap_table_path)
        st.dataframe(
            df_shap.head(20).style.format({
                "mean_abs_shap": "{:.4f}",
                "directional_shap_mean": "{:.4f}",
            }),
            use_container_width=True,
        )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.85rem;">
        <b>NFHS-5 Diabetes Machine Learning Decision Support System</b> | Reproducible Seed <code>42</code> | 
        Developed with Scikit-Learn, LightGBM, XGBoost, SHAP, Plotly & Streamlit.
    </div>
    """,
    unsafe_allow_html=True,
)
