"""
Unit tests for model pipeline, ColumnTransformer, and benchmarking components.
"""

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

from src.models.pipeline import (
    ALL_FEATURES,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_model_pipeline,
    build_preprocessor,
    get_model_catalog,
)
from src.models.train import find_optimal_threshold


def test_feature_definitions():
    """Verify feature sets are well-formed and non-overlapping."""
    assert len(NUMERIC_FEATURES) > 0
    assert len(BINARY_FEATURES) > 0
    assert len(CATEGORICAL_FEATURES) > 0

    num_set = set(NUMERIC_FEATURES)
    bin_set = set(BINARY_FEATURES)
    cat_set = set(CATEGORICAL_FEATURES)

    assert num_set.isdisjoint(bin_set), "Numeric and binary features must be disjoint"
    assert num_set.isdisjoint(cat_set), "Numeric and categorical features must be disjoint"
    assert bin_set.isdisjoint(cat_set), "Binary and categorical features must be disjoint"
    assert len(ALL_FEATURES) == len(num_set) + len(bin_set) + len(cat_set)


def test_build_preprocessor():
    """Test ColumnTransformer construction and transformation on mock dataframe."""
    preprocessor = build_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer)

    # Build mock dataframe with one sample
    data = {}
    for col in NUMERIC_FEATURES:
        data[col] = [25.0, np.nan, 40.0]
    for col in BINARY_FEATURES:
        data[col] = [0.0, 1.0, np.nan]
    for col in CATEGORICAL_FEATURES:
        data[col] = ["CategoryA", np.nan, "CategoryB"]

    mock_df = pd.DataFrame(data)
    transformed = preprocessor.fit_transform(mock_df)

    assert transformed.shape[0] == 3
    assert transformed.shape[1] >= len(NUMERIC_FEATURES) + len(BINARY_FEATURES)
    feature_names = preprocessor.get_feature_names_out()
    assert len(feature_names) == transformed.shape[1]


def test_get_model_catalog():
    """Test that model catalog contains all 4 requested algorithms and SMOTE variants."""
    catalog = get_model_catalog(scale_pos_weight=43.56, random_state=42)
    assert len(catalog) == 8

    expected_algorithms = ["Logistic Regression", "Random Forest", "LightGBM", "XGBoost"]
    for algo in expected_algorithms:
        weighted_key = f"{algo} (Class-Weighted)" if algo != "XGBoost" else f"{algo} (Scale-Pos-Weight)"
        smote_key = f"{algo} (SMOTE)"
        assert weighted_key in catalog, f"Missing weighted variant for {algo}"
        assert smote_key in catalog, f"Missing SMOTE variant for {algo}"
        assert catalog[weighted_key][1] is False, "Weighted variant should have use_smote=False"
        assert catalog[smote_key][1] is True, "SMOTE variant should have use_smote=True"


def test_build_model_pipeline():
    """Verify that end-to-end pipeline builds correctly with and without SMOTE."""
    dummy_model = LogisticRegression()

    pipe_no_smote = build_model_pipeline(dummy_model, use_smote=False)
    assert isinstance(pipe_no_smote, ImbPipeline)
    step_names = [name for name, _ in pipe_no_smote.steps]
    assert step_names == ["preprocessor", "classifier"]

    pipe_smote = build_model_pipeline(dummy_model, use_smote=True)
    assert isinstance(pipe_smote, ImbPipeline)
    step_names_smote = [name for name, _ in pipe_smote.steps]
    assert step_names_smote == ["preprocessor", "smote", "classifier"]


def test_find_optimal_threshold():
    """Verify threshold search optimization function."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.15, 0.3, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9])

    best_thresh, best_f1 = find_optimal_threshold(y_true, y_prob)
    assert 0.0 <= best_thresh <= 1.0
    assert best_f1 > 0.8
