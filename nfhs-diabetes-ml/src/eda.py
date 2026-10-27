"""
Exploratory Data Analysis (EDA) module for nfhs-diabetes-ml.
Performs descriptive analysis for demographic, socioeconomic, and anthropometric variables,
computes stratified distributions across diabetes target status,
and generates the 12 required publication-quality figures.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR, RANDOM_STATE, TABLES_DIR
from src.utils import ensure_directory, set_seed, setup_logger

logger = setup_logger(__name__)

# Consistent target-class color scheme
COLOR_NON_DIABETIC = "#2b5c8f"  # Slate Blue
COLOR_DIABETIC = "#d95f02"      # Terracotta Orange
TARGET_PALETTE = {0: COLOR_NON_DIABETIC, 1: COLOR_DIABETIC}
TARGET_LABELS = {0: "Non-Diabetic", 1: "Diabetic"}


def load_analysis_data(data_path: Optional[Union[Path, str]] = None) -> pd.DataFrame:
    """Load the processed analysis parquet dataset."""
    path = Path(data_path) if data_path else PROCESSED_DATA_DIR / "analysis_dataset.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Analysis dataset not found at: {path}")
    df = pd.read_parquet(path)
    logger.info("Loaded analysis dataset: %d rows x %d columns", df.shape[0], df.shape[1])
    return df


def calculate_numeric_summary(
    df: pd.DataFrame,
    columns: List[str],
    target_col: str = "diabetes_status",
) -> pd.DataFrame:
    """
    Calculate descriptive statistics (mean, std, median, IQR, min, max)
    overall and stratified by target class.
    """
    records = []
    for col in columns:
        if col not in df.columns:
            continue
        valid_overall = df[col].dropna()
        if len(valid_overall) == 0:
            continue

        row_overall = {
            "variable": col,
            "group": "Overall",
            "n_valid": len(valid_overall),
            "mean": valid_overall.mean(),
            "std": valid_overall.std(),
            "median": valid_overall.median(),
            "q25": valid_overall.quantile(0.25),
            "q75": valid_overall.quantile(0.75),
            "min": valid_overall.min(),
            "max": valid_overall.max(),
        }
        records.append(row_overall)

        for target_val in sorted(df[target_col].dropna().unique()):
            sub = df[df[target_col] == target_val][col].dropna()
            lbl = TARGET_LABELS.get(target_val, str(target_val))
            records.append({
                "variable": col,
                "group": lbl,
                "n_valid": len(sub),
                "mean": sub.mean(),
                "std": sub.std(),
                "median": sub.median(),
                "q25": sub.quantile(0.25),
                "q75": sub.quantile(0.75),
                "min": sub.min(),
                "max": sub.max(),
            })

    return pd.DataFrame(records)


def calculate_categorical_summary(
    df: pd.DataFrame,
    columns: List[str],
    target_col: str = "diabetes_status",
) -> pd.DataFrame:
    """
    Calculate frequency, percentage, and target-specific prevalence
    for categorical features.
    """
    records = []
    total_n = len(df)

    for col in columns:
        if col not in df.columns:
            continue
        valid_df = df.dropna(subset=[col])
        counts = valid_df[col].value_counts(dropna=False)

        for cat_val, count in counts.items():
            sub = valid_df[valid_df[col] == cat_val]
            n_diabetic = int((sub[target_col] == 1).sum())
            prev = (n_diabetic / len(sub)) * 100 if len(sub) > 0 else 0.0

            records.append({
                "variable": col,
                "category": str(cat_val),
                "count": count,
                "pct_of_sample": (count / total_n) * 100,
                "diabetic_count": n_diabetic,
                "diabetic_prevalence_pct": prev,
            })

    return pd.DataFrame(records)


def generate_all_plots(
    df: pd.DataFrame,
    output_dir: Optional[Union[Path, str]] = None,
) -> List[Path]:
    """
    Generate and save the 12 required figures with publication-quality aesthetics.
    """
    set_seed(RANDOM_STATE)
    fig_dir = Path(output_dir) if output_dir else FIGURES_DIR
    ensure_directory(fig_dir)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    saved_plots = []

    # -------------------------------------------------------------
    # Plot 1: Target-Class Count Plot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    target_counts = df["diabetes_status"].value_counts().sort_index()
    bars = ax.bar(
        [TARGET_LABELS[0], TARGET_LABELS[1]],
        target_counts.values,
        color=[COLOR_NON_DIABETIC, COLOR_DIABETIC],
        width=0.55,
        edgecolor="black",
        linewidth=0.8
    )
    ax.set_title("Diabetes Status Classification Target Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Diabetes Diagnosis Status (s728a)", fontsize=12, labelpad=8)
    ax.set_ylabel("Number of Respondents", fontsize=12, labelpad=8)
    total_resp = len(df)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:,}\n({h/total_resp*100:.2f}%)",
                    (bar.get_x() + bar.get_width() / 2., h),
                    ha="center", va="bottom", fontsize=11, fontweight="semibold",
                    xytext=(0, 4), textcoords="offset points")
    ax.set_ylim(0, max(target_counts.values) * 1.15)
    plt.tight_layout()
    p1 = fig_dir / "01_target_class_count_plot.png"
    plt.savefig(p1, dpi=200)
    plt.close()
    saved_plots.append(p1)

    # -------------------------------------------------------------
    # Plot 2: Age Histogram
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        df["age"],
        bins=15,
        kde=True,
        color=COLOR_NON_DIABETIC,
        edgecolor="black",
        linewidth=0.8,
        ax=ax
    )
    mean_age = df["age"].mean()
    median_age = df["age"].median()
    ax.axvline(mean_age, color="#d95f02", linestyle="--", linewidth=1.8, label=f"Mean: {mean_age:.1f} yrs")
    ax.axvline(median_age, color="#2ca02c", linestyle=":", linewidth=1.8, label=f"Median: {median_age:.1f} yrs")
    ax.set_title("Age Distribution of NFHS-5 Survey Respondents (Women 15-49)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Respondent Age (Years)", fontsize=12, labelpad=8)
    ax.set_ylabel("Frequency", fontsize=12, labelpad=8)
    ax.legend(fontsize=11)
    plt.tight_layout()
    p2 = fig_dir / "02_age_histogram.png"
    plt.savefig(p2, dpi=200)
    plt.close()
    saved_plots.append(p2)

    # -------------------------------------------------------------
    # Plot 3: Age Distribution by Target
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.kdeplot(
        data=df,
        x="age",
        hue="diabetes_status",
        palette=TARGET_PALETTE,
        common_norm=False,
        fill=True,
        alpha=0.35,
        linewidth=2,
        ax=ax
    )
    ax.set_title("Age Density Distribution Stratified by Diabetes Status", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Respondent Age (Years)", fontsize=12, labelpad=8)
    ax.set_ylabel("Kernel Density", fontsize=12, labelpad=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR_NON_DIABETIC, alpha=0.5),
               plt.Rectangle((0, 0), 1, 1, color=COLOR_DIABETIC, alpha=0.5)]
    ax.legend(handles, [TARGET_LABELS[0], TARGET_LABELS[1]], title="Diabetes Status", fontsize=11, title_fontsize=11)
    plt.tight_layout()
    p3 = fig_dir / "03_age_distribution_by_target.png"
    plt.savefig(p3, dpi=200)
    plt.close()
    saved_plots.append(p3)

    # -------------------------------------------------------------
    # Plot 4: BMI Histogram
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    valid_bmi = df["bmi"].dropna()
    sns.histplot(
        valid_bmi,
        bins=25,
        kde=True,
        color="#386cb0",
        edgecolor="black",
        linewidth=0.8,
        ax=ax
    )
    mean_bmi = valid_bmi.mean()
    median_bmi = valid_bmi.median()
    ax.axvline(mean_bmi, color="#d95f02", linestyle="--", linewidth=1.8, label=f"Mean: {mean_bmi:.2f} kg/m²")
    ax.axvline(median_bmi, color="#2ca02c", linestyle=":", linewidth=1.8, label=f"Median: {median_bmi:.2f} kg/m²")
    ax.axvline(25.0, color="#7570b3", linestyle="-.", linewidth=1.5, label="Overweight Cutoff (25.0)")
    ax.set_title("Body Mass Index (BMI) Distribution (Cleaned)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Body Mass Index (kg/m²)", fontsize=12, labelpad=8)
    ax.set_ylabel("Frequency", fontsize=12, labelpad=8)
    ax.legend(fontsize=10)
    plt.tight_layout()
    p4 = fig_dir / "04_bmi_histogram.png"
    plt.savefig(p4, dpi=200)
    plt.close()
    saved_plots.append(p4)

    # -------------------------------------------------------------
    # Plot 5: BMI Distribution by Target
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(
        data=df.dropna(subset=["bmi"]),
        x="diabetes_status",
        y="bmi",
        palette=[COLOR_NON_DIABETIC, COLOR_DIABETIC],
        width=0.45,
        boxprops=dict(alpha=0.85),
        showmeans=True,
        meanprops={"marker": "D", "markeredgecolor": "black", "markerfacecolor": "yellow", "markersize": "7"},
        ax=ax
    )
    ax.set_title("Body Mass Index (BMI) Stratified by Diabetes Status", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticklabels([TARGET_LABELS[0], TARGET_LABELS[1]], fontsize=11)
    ax.set_xlabel("Diabetes Diagnosis Status", fontsize=12, labelpad=8)
    ax.set_ylabel("Body Mass Index (kg/m²)", fontsize=12, labelpad=8)
    plt.tight_layout()
    p5 = fig_dir / "05_bmi_distribution_by_target.png"
    plt.savefig(p5, dpi=200)
    plt.close()
    saved_plots.append(p5)

    # -------------------------------------------------------------
    # Plot 6: BMI-Category Distribution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    bmi_order = ["Underweight (<18.5)", "Normal (18.5-24.9)", "Overweight (25.0-29.9)", "Obese (>=30.0)"]
    bmi_counts = df["bmi_category"].value_counts().reindex(bmi_order).dropna()
    bars = ax.bar(
        bmi_counts.index,
        bmi_counts.values,
        color=["#7fc97f", "#beaed4", "#fdc086", "#f0027f"],
        edgecolor="black",
        linewidth=0.8,
        width=0.55
    )
    ax.set_title("Overall Distribution of WHO BMI Categories", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("BMI Category", fontsize=12, labelpad=8)
    ax.set_ylabel("Respondent Count", fontsize=12, labelpad=8)
    n_bmi_total = bmi_counts.sum()
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:,}\n({h/n_bmi_total*100:.1f}%)",
                    (bar.get_x() + bar.get_width() / 2., h),
                    ha="center", va="bottom", fontsize=10, fontweight="semibold",
                    xytext=(0, 3), textcoords="offset points")
    ax.set_ylim(0, max(bmi_counts.values) * 1.15)
    plt.tight_layout()
    p6 = fig_dir / "06_bmi_category_distribution.png"
    plt.savefig(p6, dpi=200)
    plt.close()
    saved_plots.append(p6)

    # -------------------------------------------------------------
    # Plot 7: Target Status by BMI Category (Prevalence %)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    bmi_prev = df.groupby("bmi_category", observed=False)["diabetes_status"].mean().reindex(bmi_order) * 100
    bars = ax.bar(
        bmi_prev.index,
        bmi_prev.values,
        color="#d95f02",
        edgecolor="black",
        linewidth=0.8,
        width=0.55
    )
    ax.set_title("Observed Diabetes Prevalence Across BMI Categories", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("BMI Category", fontsize=12, labelpad=8)
    ax.set_ylabel("Observed Diabetes Prevalence (%)", fontsize=12, labelpad=8)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%",
                    (bar.get_x() + bar.get_width() / 2., h),
                    ha="center", va="bottom", fontsize=11, fontweight="semibold",
                    xytext=(0, 4), textcoords="offset points")
    ax.set_ylim(0, max(bmi_prev.values) * 1.25)
    plt.tight_layout()
    p7 = fig_dir / "07_target_status_by_bmi_category.png"
    plt.savefig(p7, dpi=200)
    plt.close()
    saved_plots.append(p7)

    # -------------------------------------------------------------
    # Plot 8: Target Status by Residence
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    res_prev = df.groupby("residence", observed=False)["diabetes_status"].mean() * 100
    res_order = ["Urban", "Rural"]
    bars = ax.bar(
        res_order,
        [res_prev.get(r, 0) for r in res_order],
        color=["#2b5c8f", "#7fc97f"],
        edgecolor="black",
        linewidth=0.8,
        width=0.5
    )
    ax.set_title("Observed Diabetes Prevalence by Place of Residence", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Place of Residence", fontsize=12, labelpad=8)
    ax.set_ylabel("Observed Diabetes Prevalence (%)", fontsize=12, labelpad=8)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%",
                    (bar.get_x() + bar.get_width() / 2., h),
                    ha="center", va="bottom", fontsize=11, fontweight="semibold",
                    xytext=(0, 4), textcoords="offset points")
    ax.set_ylim(0, max(res_prev.values) * 1.3)
    plt.tight_layout()
    p8 = fig_dir / "08_target_status_by_residence.png"
    plt.savefig(p8, dpi=200)
    plt.close()
    saved_plots.append(p8)

    # -------------------------------------------------------------
    # Plot 9: Target Status by Education
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    edu_order = ["No education", "Primary", "Secondary", "Higher"]
    edu_prev = df.groupby("education", observed=False)["diabetes_status"].mean().reindex(edu_order) * 100
    bars = ax.bar(
        edu_prev.index,
        edu_prev.values,
        color="#386cb0",
        edgecolor="black",
        linewidth=0.8,
        width=0.55
    )
    ax.set_title("Observed Diabetes Prevalence by Highest Educational Attainment", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Educational Level", fontsize=12, labelpad=8)
    ax.set_ylabel("Observed Diabetes Prevalence (%)", fontsize=12, labelpad=8)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%",
                    (bar.get_x() + bar.get_width() / 2., h),
                    ha="center", va="bottom", fontsize=11, fontweight="semibold",
                    xytext=(0, 4), textcoords="offset points")
    ax.set_ylim(0, max(edu_prev.values) * 1.25)
    plt.tight_layout()
    p9 = fig_dir / "09_target_status_by_education.png"
    plt.savefig(p9, dpi=200)
    plt.close()
    saved_plots.append(p9)

    # -------------------------------------------------------------
    # Plot 10: Target Status by Wealth Group
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    wealth_order = ["Poorest", "Poorer", "Middle", "Richer", "Richest"]
    wealth_prev = df.groupby("wealth_group", observed=False)["diabetes_status"].mean().reindex(wealth_order) * 100
    bars = ax.bar(
        wealth_prev.index,
        wealth_prev.values,
        color="#2ca02c",
        edgecolor="black",
        linewidth=0.8,
        width=0.55
    )
    ax.set_title("Observed Diabetes Prevalence Across Wealth Quintiles", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Household Wealth Index Quintile", fontsize=12, labelpad=8)
    ax.set_ylabel("Observed Diabetes Prevalence (%)", fontsize=12, labelpad=8)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%",
                    (bar.get_x() + bar.get_width() / 2., h),
                    ha="center", va="bottom", fontsize=11, fontweight="semibold",
                    xytext=(0, 4), textcoords="offset points")
    ax.set_ylim(0, max(wealth_prev.values) * 1.25)
    plt.tight_layout()
    p10 = fig_dir / "10_target_status_by_wealth_group.png"
    plt.savefig(p10, dpi=200)
    plt.close()
    saved_plots.append(p10)

    # -------------------------------------------------------------
    # Plot 11: Target Status by Age Group
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    age_order = ["15-24", "25-34", "35-44", "45-49"]
    age_prev = df.groupby("age_group", observed=False)["diabetes_status"].mean().reindex(age_order) * 100
    bars = ax.bar(
        age_prev.index,
        age_prev.values,
        color="#7570b3",
        edgecolor="black",
        linewidth=0.8,
        width=0.55
    )
    ax.set_title("Observed Diabetes Prevalence Across Age Groups", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Age Bracket (Years)", fontsize=12, labelpad=8)
    ax.set_ylabel("Observed Diabetes Prevalence (%)", fontsize=12, labelpad=8)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%",
                    (bar.get_x() + bar.get_width() / 2., h),
                    ha="center", va="bottom", fontsize=11, fontweight="semibold",
                    xytext=(0, 4), textcoords="offset points")
    ax.set_ylim(0, max(age_prev.values) * 1.25)
    plt.tight_layout()
    p11 = fig_dir / "11_target_status_by_age_group.png"
    plt.savefig(p11, dpi=200)
    plt.close()
    saved_plots.append(p11)

    # -------------------------------------------------------------
    # Plot 12: Target Status by Region (6 Administrative Regions)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    region_prev = df.groupby("region", observed=False)["diabetes_status"].mean().sort_values(ascending=False) * 100
    bars = ax.bar(
        region_prev.index,
        region_prev.values,
        color="#e7298a",
        edgecolor="black",
        linewidth=0.8,
        width=0.55
    )
    ax.set_title("Observed Diabetes Prevalence Across Indian Geographic Regions", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Geographic Region", fontsize=12, labelpad=8)
    ax.set_ylabel("Observed Diabetes Prevalence (%)", fontsize=12, labelpad=8)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%",
                    (bar.get_x() + bar.get_width() / 2., h),
                    ha="center", va="bottom", fontsize=11, fontweight="semibold",
                    xytext=(0, 4), textcoords="offset points")
    ax.set_ylim(0, max(region_prev.values) * 1.25)
    plt.tight_layout()
    p12 = fig_dir / "12_target_status_by_region.png"
    plt.savefig(p12, dpi=200)
    plt.close()
    saved_plots.append(p12)

    logger.info("Successfully generated and saved all 12 plots to: %s", fig_dir)
    return saved_plots


def run_full_eda(
    data_path: Optional[Union[Path, str]] = None,
    output_dir: Optional[Union[Path, str]] = None,
) -> Tuple[List[Path], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute complete EDA pipeline and save figures and tables."""
    df = load_analysis_data(data_path)
    plots = generate_all_plots(df, output_dir=output_dir)

    ensure_directory(TABLES_DIR)
    # Demographic & Socioeconomic summaries
    cat_cols = ["age_group", "residence", "marital_status", "education", "wealth_group", "occupation", "health_insurance", "region", "bmi_category"]
    cat_summary = calculate_categorical_summary(df, cat_cols)
    cat_summary.to_csv(TABLES_DIR / "eda_categorical_summary.csv", index=False)

    num_cols = ["age", "height_cm", "weight_kg", "bmi", "systolic_bp", "diastolic_bp"]
    num_summary = calculate_numeric_summary(df, num_cols)
    num_summary.to_csv(TABLES_DIR / "eda_numeric_summary.csv", index=False)

    # Prevalence table
    logger.info("Saved summary tables to: %s", TABLES_DIR)
    return plots, cat_summary, num_summary, df


if __name__ == "__main__":
    run_full_eda()
