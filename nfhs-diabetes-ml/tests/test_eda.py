"""
Unit tests for preprocessing and exploratory data analysis modules.
"""

from pathlib import Path
import pytest
import pandas as pd

from src.preprocessing import build_analysis_dataset
from src.eda import load_analysis_data, calculate_numeric_summary, calculate_categorical_summary


def test_analysis_dataset_exists_and_valid():
    df = load_analysis_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 11898
    assert "diabetes_status" in df.columns
    assert "s728b" not in df.columns  # Ensure leakage variable was dropped
    assert set(df["diabetes_status"].unique()) == {0, 1}


def test_eda_summaries():
    df = load_analysis_data()
    cat_summary = calculate_categorical_summary(df, ["age_group", "residence", "wealth_group"])
    assert not cat_summary.empty
    assert "diabetic_prevalence_pct" in cat_summary.columns

    num_summary = calculate_numeric_summary(df, ["age", "bmi"])
    assert not num_summary.empty
    assert "mean" in num_summary.columns
    assert "median" in num_summary.columns
