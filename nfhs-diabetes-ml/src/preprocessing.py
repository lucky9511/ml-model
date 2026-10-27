"""
Preprocessing module for nfhs-diabetes-ml.
Loads raw Stata data, cleans sentinel codes, rescales physiological measurements,
encodes categorical attributes, creates demographic/anthropometric groups,
and persists data/processed/analysis_dataset.parquet.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR, RANDOM_STATE
from src.data_loader import load_raw_data, replace_special_codes_with_nan
from src.utils import ensure_directory, set_seed, setup_logger

logger = setup_logger(__name__)

# State code mapping for NFHS-5 (v024: 1 to 36)
STATE_MAP = {
    1: "Jammu & Kashmir", 2: "Himachal Pradesh", 3: "Punjab", 4: "Chandigarh",
    5: "Uttarakhand", 6: "Haryana", 7: "NCT of Delhi", 8: "Rajasthan",
    9: "Uttar Pradesh", 10: "Bihar", 11: "Sikkim", 12: "Arunachal Pradesh",
    13: "Nagaland", 14: "Manipur", 15: "Mizoram", 16: "Tripura",
    17: "Meghalaya", 18: "Assam", 19: "West Bengal", 20: "Jharkhand",
    21: "Odisha", 22: "Chhattisgarh", 23: "Madhya Pradesh", 24: "Gujarat",
    25: "Dadra & Nagar Haveli and Daman & Diu", 26: "Maharashtra", 27: "Andhra Pradesh",
    28: "Karnataka", 29: "Goa", 30: "Lakshadweep", 31: "Kerala",
    32: "Tamil Nadu", 33: "Puducherry", 34: "Andaman & Nicobar Islands",
    35: "Telangana", 36: "Ladakh"
}

# Regional groupings across Indian States & UTs
REGION_MAP = {
    "Jammu & Kashmir": "North", "Himachal Pradesh": "North", "Punjab": "North",
    "Chandigarh": "North", "Uttarakhand": "North", "Haryana": "North",
    "NCT of Delhi": "North", "Rajasthan": "North", "Ladakh": "North",
    "Uttar Pradesh": "Central", "Madhya Pradesh": "Central", "Chhattisgarh": "Central",
    "Bihar": "East", "Jharkhand": "East", "Odisha": "East", "West Bengal": "East",
    "Sikkim": "Northeast", "Arunachal Pradesh": "Northeast", "Nagaland": "Northeast",
    "Manipur": "Northeast", "Mizoram": "Northeast", "Tripura": "Northeast",
    "Meghalaya": "Northeast", "Assam": "Northeast",
    "Gujarat": "West", "Maharashtra": "West", "Goa": "West",
    "Dadra & Nagar Haveli and Daman & Diu": "West",
    "Andhra Pradesh": "South", "Karnataka": "South", "Kerala": "South",
    "Tamil Nadu": "South", "Puducherry": "South", "Lakshadweep": "South",
    "Andaman & Nicobar Islands": "South", "Telangana": "South"
}


def build_analysis_dataset(
    raw_file_path: Optional[Union[Path, str]] = None,
    output_path: Optional[Union[Path, str]] = None,
) -> pd.DataFrame:
    """
    Transform raw NFHS-5 data into the cleaned analysis dataset.

    Parameters
    ----------
    raw_file_path : Path or str, optional
        Path to raw .dta file. If None, auto-discovered.
    output_path : Path or str, optional
        Destination path for analysis_dataset.parquet.

    Returns
    -------
    pd.DataFrame
        Cleaned and feature-engineered analysis dataset.
    """
    set_seed(RANDOM_STATE)
    df_raw, meta = load_raw_data(file_path=raw_file_path)
    logger.info("Raw input shape: %s", df_raw.shape)

    # 1. Target Definition: Filter out indeterminate responses (8: Don't know, 9: Missing)
    # Target: s728a (Told by doctor: Diabetes)
    valid_target_mask = df_raw["s728a"].isin([0, 1])
    logger.info("Excluded %d rows with indeterminate target status (8/9)", (~valid_target_mask).sum())
    df = df_raw[valid_target_mask].copy().reset_index(drop=True)

    # Target variable mapping
    df["diabetes_status"] = df["s728a"].astype(int)
    df["target"] = df["diabetes_status"]  # convenient alias

    # 2. Prevent Data Leakage: Drop s728b (taking medication for diabetes)
    if "s728b" in df.columns:
        logger.info("Dropping post-diagnostic treatment feature 's728b' to avoid target leakage")
        df = df.drop(columns=["s728b"])

    # 3. Clean Sentinel Missing Codes
    df = replace_special_codes_with_nan(df)

    # 4. Rescale Anthropometric & Physiological Measurements
    # v438: Height in tenths of cm -> cm
    df["height_cm"] = df["v438"] / 10.0

    # v447: Weight in tenths of kg -> kg
    df["weight_kg"] = df["v447"] / 10.0

    # v445: BMI in hundredths -> kg/m2
    df["bmi"] = df["v445"] / 100.0

    # Blood Pressure (sb221: SBP, sb222: DBP)
    df["systolic_bp"] = df["sb221"]
    df["diastolic_bp"] = df["sb222"]

    # Survey sample weight (v005: 6 implied decimals)
    df["sample_weight"] = df["v005"] / 1_000_000.0

    # 5. Demographic Feature Engineering
    df["age"] = df["v012"].astype(int)
    df["age_group"] = pd.cut(
        df["age"],
        bins=[14, 24, 34, 44, 50],
        labels=["15-24", "25-34", "35-44", "45-49"],
        right=True
    ).astype(str)

    # Residence (v025: 1=Urban, 2=Rural)
    residence_labels = {1: "Urban", 2: "Rural"}
    df["residence"] = df["v025"].map(residence_labels)

    # Marital status (v502: 0=Never married, 1=Currently married, 2=Formerly married)
    marital_labels = {0: "Never married", 1: "Currently married", 2: "Formerly married"}
    df["marital_status"] = df["v502"].map(marital_labels)

    # State & Region mapping
    df["state"] = df["v024"].map(STATE_MAP).fillna("Unknown State")
    df["region"] = df["state"].map(REGION_MAP).fillna("Other")

    # 6. Socioeconomic Feature Engineering
    # Education level (v106: 0=No education, 1=Primary, 2=Secondary, 3=Higher)
    edu_labels = {0: "No education", 1: "Primary", 2: "Secondary", 3: "Higher"}
    df["education"] = df["v106"].map(edu_labels)

    # Wealth index (v190: 1=Poorest to 5=Richest)
    wealth_labels = {1: "Poorest", 2: "Poorer", 3: "Middle", 4: "Richer", 5: "Richest"}
    df["wealth_group"] = df["v190"].map(wealth_labels)

    # Occupation / working status (v714: 0=Not working, 1=Currently working)
    df["occupation"] = df["v714"].map({0: "Not working", 1: "Currently working"})

    # Health insurance coverage (v481: 0=Not covered, 1=Covered)
    df["health_insurance"] = df["v481"].map({0: "Not covered", 1: "Covered"})

    # 7. Anthropometric Categorization (WHO Global & Asian BMI Cutoffs)
    def assign_bmi_category(val: float) -> Optional[str]:
        if pd.isna(val):
            return None
        if val < 18.5:
            return "Underweight (<18.5)"
        elif val < 25.0:
            return "Normal (18.5-24.9)"
        elif val < 30.0:
            return "Overweight (25.0-29.9)"
        else:
            return "Obese (>=30.0)"

    df["bmi_category"] = df["bmi"].apply(assign_bmi_category)

    # Note: Waist circumference and Hip circumference are absent from the individual recode dataset
    df["waist_circumference"] = np.nan
    df["hip_circumference"] = np.nan
    df["waist_hip_ratio"] = np.nan

    # 8. Persist Analysis Dataset
    dest = Path(output_path) if output_path else PROCESSED_DATA_DIR / "analysis_dataset.parquet"
    ensure_directory(dest.parent)
    df.to_parquet(dest, index=False)
    logger.info("Successfully saved analysis dataset to: %s (%d rows, %d columns)", dest, df.shape[0], df.shape[1])

    return df


if __name__ == "__main__":
    build_analysis_dataset()
