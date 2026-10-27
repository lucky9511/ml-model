"""
Unit tests for SHAP explainability module.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.explainability import (
    clean_feature_name,
    build_shap_explainer,
    explain_dataset,
    compute_feature_importance_table,
)
from src.models.pipeline import ALL_FEATURES
from src.utils import load_artifact


def test_clean_feature_name():
    """Verify feature cleaning and clinical beautification."""
    assert clean_feature_name("num__age") == "Age (Years)"
    assert clean_feature_name("num__bmi") == "Body Mass Index (kg/m²)"
    assert clean_feature_name("bin__s728c") == "Hypertension Diagnosis"
    assert clean_feature_name("cat__wealth_group_Richest") == "Wealth Quintile: Richest"
    assert clean_feature_name("unknown_feature") == "unknown_feature"


def test_shap_explainer_and_explanation():
    """Verify SHAP explainer builds and computes valid Shapley values."""
    model_path = MODELS_DIR / "best_model.joblib"
    assert model_path.exists(), "Trained best_model.joblib must exist"

    pipeline = load_artifact(model_path)
    df = pd.read_parquet(PROCESSED_DATA_DIR / "analysis_dataset.parquet")
    X = df[ALL_FEATURES].head(50)

    explainer, raw_names, clean_names = build_shap_explainer(pipeline, X, max_background_samples=30)
    assert len(clean_names) == 53
    assert explainer is not None

    shap_values, names = explain_dataset(pipeline, X, sample_size=20)
    assert shap_values.shape[0] == 20
    assert shap_values.shape[1] == 53
    assert not np.isnan(shap_values.values).any()


def test_feature_importance_table(tmp_path):
    """Verify feature importance ranking and export."""
    model_path = MODELS_DIR / "best_model.joblib"
    pipeline = load_artifact(model_path)
    df = pd.read_parquet(PROCESSED_DATA_DIR / "analysis_dataset.parquet")
    X = df[ALL_FEATURES].head(30)

    shap_values, _ = explain_dataset(pipeline, X, sample_size=20)
    temp_csv = tmp_path / "test_shap_table.csv"
    table = compute_feature_importance_table(shap_values, temp_csv)

    assert temp_csv.exists()
    assert len(table) == 53
    assert "mean_abs_shap" in table.columns
    assert "rank" in table.columns
    # Check that table is sorted descending
    assert table["mean_abs_shap"].is_monotonic_decreasing
