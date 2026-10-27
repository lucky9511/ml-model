"""
Data loader module for nfhs-diabetes-ml.
Loads Stata (.dta) datasets using pyreadstat, discovers files, and extracts survey metadata.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import pyreadstat

from src.config import PROJECT_ROOT, RAW_DATA_DIR
from src.utils import setup_logger

logger = setup_logger(__name__)

# Standard NFHS/DHS special sentinel codes mapping
DEFAULT_SPECIAL_CODES: Dict[str, List[Union[int, float]]] = {
    # Anthropometry: 9998 = Flagged / out of range / refused / not measured
    "v438": [9998, 9999],
    "v445": [9998, 9999],
    "v447": [9998, 9999],
    # Blood pressure: 998 = Not measured / refused
    "sb221": [998, 999],
    "sb222": [998, 999],
    # Diagnosed conditions & behaviors: 8 = Don't know, 9 = Missing
    "s728a": [8, 9],
    "s728b": [9],
    "s728c": [9],
    "v463a": [9],
    "s716": [9],
    "v714": [9],
    "v481": [9],
    "v131": [8],  # 8 = Don't know for caste
}

# Add diet frequency variables: 9 = Missing
for _code in ["s706a", "s706b", "s706c", "s706d", "s706e", "s706f", "s706g", "s706h", "s706i"]:
    DEFAULT_SPECIAL_CODES[_code] = [9]


def find_dta_file(
    search_dirs: Optional[List[Union[Path, str]]] = None,
    preferred_pattern: str = "*.dta",
) -> Path:
    """
    Search for an NFHS-5 Stata (.dta) file across candidate directories.
    Does not assume a fixed filename or single location.

    Parameters
    ----------
    search_dirs : list of (Path or str), optional
        Directories to search in. Defaults to [RAW_DATA_DIR, PROJECT_ROOT, parent directories].
    preferred_pattern : str, default="*.dta"
        Glob pattern for matching Stata files.

    Returns
    -------
    Path
        Absolute path to the first matched .dta file.

    Raises
    ------
    FileNotFoundError
        If no .dta file could be found in candidate directories.
    """
    if search_dirs is None:
        search_dirs = [
            RAW_DATA_DIR,
            PROJECT_ROOT,
            PROJECT_ROOT.parent,
            PROJECT_ROOT.parent / "ml project2.0" / "data" / "raw",
        ]

    for directory in search_dirs:
        dir_path = Path(directory)
        if not dir_path.exists():
            continue

        # Check non-recursive first in the directory
        dta_files = list(dir_path.glob(preferred_pattern))
        if not dta_files:
            # Case-insensitive / upper-case check
            dta_files = list(dir_path.glob("*.DTA"))
        if not dta_files:
            # Recursive search in subdirectories
            dta_files = list(dir_path.rglob(preferred_pattern))
            if not dta_files:
                dta_files = list(dir_path.rglob("*.DTA"))

        if dta_files:
            found = dta_files[0].resolve()
            logger.info("Located NFHS-5 DTA file at: %s", found)
            return found

    raise FileNotFoundError(
        f"No Stata (.dta) file found across search directories: {[str(d) for d in search_dirs]}"
    )


def load_raw_data(
    file_path: Optional[Union[Path, str]] = None,
    clean_special_codes: bool = False,
    special_codes_map: Optional[Dict[str, List[Union[int, float]]]] = None,
) -> Tuple[pd.DataFrame, Any]:
    """
    Load an NFHS-5 Stata file using pyreadstat.

    Parameters
    ----------
    file_path : Path or str, optional
        Path to the .dta file. If None, find_dta_file() is invoked automatically.
    clean_special_codes : bool, default=False
        If True, replaces sentinel missing values (e.g., 9998, 998, 9) with np.nan.
    special_codes_map : dict, optional
        Custom mapping of column names to sentinel values to treat as NaN.

    Returns
    -------
    df : pd.DataFrame
        Loaded pandas DataFrame.
    meta : pyreadstat.metadata_container
        Metadata container containing variable labels, value labels, and column names.
    """
    if file_path is None:
        target_path = find_dta_file()
    else:
        target_path = Path(file_path).resolve()
        if not target_path.is_file():
            raise FileNotFoundError(f"Specified data file does not exist: {target_path}")

    logger.info("Loading Stata dataset using pyreadstat from: %s", target_path)
    df, meta = pyreadstat.read_dta(str(target_path))
    logger.info("Loaded dataset successfully: %d rows, %d columns", df.shape[0], df.shape[1])

    if clean_special_codes:
        df = replace_special_codes_with_nan(df, special_codes_map=special_codes_map)

    return df, meta


def replace_special_codes_with_nan(
    df: pd.DataFrame,
    special_codes_map: Optional[Dict[str, List[Union[int, float]]]] = None,
) -> pd.DataFrame:
    """
    Replace survey sentinel codes (e.g. 9998, 998, 9) with np.nan.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    special_codes_map : dict, optional
        Mapping of column name to sentinel code values. Defaults to DEFAULT_SPECIAL_CODES.

    Returns
    -------
    pd.DataFrame
        DataFrame with sentinel codes replaced by np.nan.
    """
    cleaned_df = df.copy()
    mapping = special_codes_map or DEFAULT_SPECIAL_CODES

    for col, codes in mapping.items():
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].replace(codes, np.nan)

    return cleaned_df


def decode_categoricals(
    df: pd.DataFrame,
    meta: Any,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Create a copy of DataFrame with numeric coded categories replaced by human-readable
    value label strings from pyreadstat metadata.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or cleaned DataFrame.
    meta : pyreadstat metadata container
        Metadata object from pyreadstat.
    columns : list of str, optional
        Specific columns to decode. If None, all columns with value labels are decoded.

    Returns
    -------
    pd.DataFrame
        DataFrame with decoded categorical strings.
    """
    decoded_df = df.copy()
    var_to_label = meta.variable_to_label or {}
    value_labels = meta.value_labels or {}

    cols_to_process = columns if columns is not None else list(var_to_label.keys())

    for col in cols_to_process:
        if col in decoded_df.columns and col in var_to_label:
            label_name = var_to_label[col]
            if label_name in value_labels:
                vmap = value_labels[label_name]
                decoded_df[col] = decoded_df[col].map(vmap).fillna(decoded_df[col])

    return decoded_df
