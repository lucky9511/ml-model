"""
Data dictionary and survey variable inspection module for nfhs-diabetes-ml.
Builds variable inventories, missingness reports, and domain keyword mapping for NFHS-5 data.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.data_loader import DEFAULT_SPECIAL_CODES, load_raw_data
from src.utils import setup_logger

logger = setup_logger(__name__)

# Mandatory keywords to search across variable names and labels
SEARCH_DOMAINS: Dict[str, List[str]] = {
    "Diabetes": ["diabetes", "sugar", "glucose", "insulin", "s728a", "s728b"],
    "Blood pressure": ["blood pressure", "sbp", "dbp", "sb221", "sb222"],
    "Hypertension": ["hypertension", "high blood pressure", "s728c"],
    "BMI": ["bmi", "body mass index", "v445"],
    "Height": ["height", "v438"],
    "Weight": ["weight", "v447"],
    "Waist circumference": ["waist"],
    "Hip circumference": ["hip"],
    "Age": ["age", "v012"],
    "Residence": ["residence", "urban", "rural", "v025"],
    "State or region": ["state", "region", "territory", "v024"],
    "Education": ["education", "literacy", "school", "v106"],
    "Occupation": ["occupation", "work", "employment", "v714"],
    "Wealth index": ["wealth", "quintile", "asset", "v190"],
    "Health insurance": ["insurance", "covered", "v481"],
    "Tobacco": ["tobacco", "cigarette", "bidi", "smoking", "v463a"],
    "Alcohol": ["alcohol", "drink", "liquor", "s716"],
    "Diet": ["diet", "food", "milk", "pulses", "fruit", "egg", "fish", "meat", "fried", "aerated", "s706"],
    "Healthcare access": ["healthcare", "hospital", "doctor", "facility", "insurance", "v481"],
    "Survey weights": ["weight", "sample weight", "v005"],
    "PSU": ["psu", "cluster", "primary sampling unit", "v001"],
    "Strata": ["strata", "stratum", "sample strata", "v022", "v023"],
}


def build_data_inventory(df: pd.DataFrame, meta: Any) -> pd.DataFrame:
    """
    Construct a comprehensive data inventory table for all variables in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset loaded from Stata file.
    meta : pyreadstat metadata container
        Metadata from pyreadstat.

    Returns
    -------
    pd.DataFrame
        Inventory table with columns: [variable, label, dtype, non_null_count,
        unique_values, min_val, max_val, special_codes_detected]
    """
    records = []
    col_names = meta.column_names if hasattr(meta, "column_names") else df.columns.tolist()
    labels_map = meta.column_names_to_labels if hasattr(meta, "column_names_to_labels") else {}

    for col in col_names:
        series = df[col]
        label = labels_map.get(col, "")
        dtype = str(series.dtype)
        non_null = int(series.notna().sum())
        n_unique = int(series.nunique())

        # Determine min and max
        if pd.api.types.is_numeric_dtype(series):
            min_val = float(series.min()) if non_null > 0 else np.nan
            max_val = float(series.max()) if non_null > 0 else np.nan
        else:
            min_val = str(series.min()) if non_null > 0 else ""
            max_val = str(series.max()) if non_null > 0 else ""

        # Check detected special codes
        detected_specials = []
        if col in DEFAULT_SPECIAL_CODES:
            for code in DEFAULT_SPECIAL_CODES[col]:
                count = int((series == code).sum())
                if count > 0:
                    detected_specials.append(f"{code} (n={count})")

        records.append({
            "variable": col,
            "label": label,
            "dtype": dtype,
            "non_null_count": non_null,
            "unique_values": n_unique,
            "min_val": min_val,
            "max_val": max_val,
            "special_codes_detected": "; ".join(detected_specials) if detected_specials else "None",
        })

    return pd.DataFrame(records)


def build_variable_dictionary(df: pd.DataFrame, meta: Any) -> pd.DataFrame:
    """
    Construct an enriched variable dictionary with domain category, value labels,
    scaling factors, and notes.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset DataFrame.
    meta : pyreadstat metadata container
        pyreadstat metadata.

    Returns
    -------
    pd.DataFrame
        Dictionary DataFrame with detailed schema details.
    """
    records = []
    col_names = meta.column_names if hasattr(meta, "column_names") else df.columns.tolist()
    labels_map = meta.column_names_to_labels if hasattr(meta, "column_names_to_labels") else {}
    var_to_vlab = meta.variable_to_label if hasattr(meta, "variable_to_label") else {}
    val_labs = meta.value_labels if hasattr(meta, "value_labels") else {}

    # Category heuristic
    def assign_category(var_name: str, var_label: str) -> str:
        name_lower = var_name.lower()
        lbl_lower = var_label.lower()
        if name_lower in ["caseid", "v001", "v002", "v003", "v005"]:
            return "Survey Design & Identifiers"
        if name_lower in ["v012", "v024", "v025", "v106", "v130", "v131", "v190", "v502", "v714"]:
            return "Demographic & Socioeconomic"
        if name_lower in ["v438", "v445", "v447"]:
            return "Anthropometric Biomarkers"
        if name_lower in ["sb221", "sb222", "s728c"]:
            return "Blood Pressure & Hypertension"
        if name_lower in ["v463a", "s716"]:
            return "Lifestyle & Substance Use"
        if name_lower.startswith("s706"):
            return "Dietary Frequency"
        if name_lower in ["s728a", "s728b"]:
            return "Diabetes Status & Treatment"
        if name_lower in ["s728d", "s728e", "s728f", "s728g"]:
            return "Chronic Comorbidities"
        if name_lower == "v481":
            return "Healthcare Access & Insurance"
        return "Other"

    # Scale / transformation notes
    scale_notes = {
        "v005": "Divided by 1,000,000 for normalized survey sampling weight",
        "v438": "Height in mm/tenths of cm (divide by 10 for cm). 9998 = Flagged/Refused",
        "v445": "BMI with 2 implied decimals (divide by 100 for kg/m2). 9998 = Flagged/Refused",
        "v447": "Weight with 1 implied decimal (divide by 10 for kg). 9998 = Flagged/Refused",
        "sb221": "Systolic blood pressure (mmHg). 998 = Not measured/Refused",
        "sb222": "Diastolic blood pressure (mmHg). 998 = Not measured/Refused",
        "s728a": "0=No, 1=Yes, 8=Don't know, 9=Missing",
        "s728b": "0=No, 1=Yes, 9=Missing (CAUTION: Treatment leakage if used as predictor)",
    }

    for col in col_names:
        label = labels_map.get(col, "")
        cat = assign_category(col, label)
        vlab_name = var_to_vlab.get(col)
        vlab_dict = val_labs.get(vlab_name, {}) if vlab_name else {}
        vlab_str = "; ".join([f"{k}: {v}" for k, v in vlab_dict.items()]) if vlab_dict else "Continuous / Numeric ID"
        notes = scale_notes.get(col, "Standard coded integer")

        special_codes = DEFAULT_SPECIAL_CODES.get(col, [])
        special_str = ", ".join([str(c) for c in special_codes]) if special_codes else "None"

        records.append({
            "variable": col,
            "variable_label": label,
            "domain_category": cat,
            "dtype": str(df[col].dtype),
            "value_labels": vlab_str,
            "special_missing_codes": special_str,
            "measurement_scale_notes": notes,
        })

    return pd.DataFrame(records)


def build_missingness_report(df: pd.DataFrame, meta: Any) -> pd.DataFrame:
    """
    Calculate raw missingness (NaNs), special sentinel code counts, and effective
    total missing rates for every variable.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset DataFrame.
    meta : pyreadstat metadata container
        pyreadstat metadata.

    Returns
    -------
    pd.DataFrame
        Report table with columns: [variable, variable_label, total_rows, raw_nan_count,
        sentinel_missing_count, effective_missing_count, effective_missing_pct,
        special_codes_breakdown]
    """
    records = []
    total_rows = len(df)
    labels_map = meta.column_names_to_labels if hasattr(meta, "column_names_to_labels") else {}

    for col in df.columns:
        label = labels_map.get(col, "")
        raw_nan = int(df[col].isna().sum())
        sentinel_count = 0
        breakdown_items = []

        if col in DEFAULT_SPECIAL_CODES:
            for code in DEFAULT_SPECIAL_CODES[col]:
                c_count = int((df[col] == code).sum())
                if c_count > 0:
                    sentinel_count += c_count
                    breakdown_items.append(f"code_{code}={c_count}")

        effective_missing = raw_nan + sentinel_count
        effective_pct = round((effective_missing / total_rows) * 100, 2)

        records.append({
            "variable": col,
            "variable_label": label,
            "total_rows": total_rows,
            "raw_nan_count": raw_nan,
            "sentinel_missing_count": sentinel_count,
            "effective_missing_count": effective_missing,
            "effective_missing_pct": effective_pct,
            "special_codes_breakdown": "; ".join(breakdown_items) if breakdown_items else "None",
        })

    return pd.DataFrame(records)


def search_domains(df: pd.DataFrame, meta: Any) -> Dict[str, List[Dict[str, Any]]]:
    """
    Search variable names and labels across the 21 required domain keywords.
    Uses regex word boundary matching to prevent substring collisions (e.g. 'age' in 'leakage').

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    meta : pyreadstat metadata container
        Metadata container.

    Returns
    -------
    dict
        Dictionary mapping domain names to lists of matched variable summaries.
    """
    import re

    labels_map = meta.column_names_to_labels if hasattr(meta, "column_names_to_labels") else {}
    results: Dict[str, List[Dict[str, Any]]] = {}

    for domain, keywords in SEARCH_DOMAINS.items():
        matched = []
        for col in df.columns:
            col_str = col.lower()
            lbl_str = labels_map.get(col, "").lower()
            combined_text = f"{col_str} {lbl_str}"

            match_found = False
            for kw in keywords:
                # Use word boundary search
                pattern = r"\b" + re.escape(kw.lower()) + r"\b"
                if re.search(pattern, combined_text):
                    match_found = True
                    break

            if match_found:
                matched.append({
                    "variable": col,
                    "label": labels_map.get(col, ""),
                    "dtype": str(df[col].dtype),
                })

        results[domain] = matched

    return results


def generate_all_reports(
    output_dir: Optional[Union[Path, str]] = None,
    data_file: Optional[Union[Path, str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate and save data_inventory.csv, variable_dictionary.csv, and missingness_report.csv.

    Parameters
    ----------
    output_dir : Path or str, optional
        Destination directory. Defaults to reports/.
    data_file : Path or str, optional
        Path to .dta file. If None, automatically located.

    Returns
    -------
    tuple of (inventory_df, dictionary_df, missingness_df)
    """
    from src.config import REPORTS_DIR

    target_dir = Path(output_dir) if output_dir else REPORTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    df, meta = load_raw_data(file_path=data_file)

    inventory_df = build_data_inventory(df, meta)
    dictionary_df = build_variable_dictionary(df, meta)
    missingness_df = build_missingness_report(df, meta)

    inv_path = target_dir / "data_inventory.csv"
    dict_path = target_dir / "variable_dictionary.csv"
    miss_path = target_dir / "missingness_report.csv"

    inventory_df.to_csv(inv_path, index=False)
    dictionary_df.to_csv(dict_path, index=False)
    missingness_df.to_csv(miss_path, index=False)

    logger.info("Saved data inventory to: %s", inv_path)
    logger.info("Saved variable dictionary to: %s", dict_path)
    logger.info("Saved missingness report to: %s", miss_path)

    return inventory_df, dictionary_df, missingness_df


if __name__ == "__main__":
    generate_all_reports()
