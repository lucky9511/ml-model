"""
Unit tests for data loader and data dictionary modules.
"""

import pytest
import pandas as pd
from pathlib import Path

from src.data_loader import find_dta_file, load_raw_data, decode_categoricals
from src.data_dictionary import (
    build_data_inventory,
    build_variable_dictionary,
    build_missingness_report,
    search_domains,
)


def test_find_dta_file():
    path = find_dta_file()
    assert path.exists()
    assert path.suffix.lower() == ".dta"


def test_load_raw_data():
    df, meta = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 12000
    assert df.shape[1] == 38
    assert "s728a" in df.columns
    assert "s728b" in df.columns
    assert hasattr(meta, "column_names")
    assert hasattr(meta, "value_labels")


def test_reports_generation():
    df, meta = load_raw_data()
    inv_df = build_data_inventory(df, meta)
    assert len(inv_df) == 38
    assert "variable" in inv_df.columns
    assert "special_codes_detected" in inv_df.columns

    dict_df = build_variable_dictionary(df, meta)
    assert len(dict_df) == 38
    assert "domain_category" in dict_df.columns

    miss_df = build_missingness_report(df, meta)
    assert len(miss_df) == 38
    assert "effective_missing_pct" in miss_df.columns


def test_domain_search():
    df, meta = load_raw_data()
    results = search_domains(df, meta)
    assert "Diabetes" in results
    diabetes_vars = [item["variable"] for item in results["Diabetes"]]
    assert "s728a" in diabetes_vars
    assert "s728b" in diabetes_vars
