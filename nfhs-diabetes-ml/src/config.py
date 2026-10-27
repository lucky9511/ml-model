"""
Project configuration module for nfhs-diabetes-ml.
Loads YAML configurations and provides central path definitions and constants.
"""

from pathlib import Path
from typing import Any, Dict
import yaml

# Central reproducibility constant
RANDOM_STATE: int = 42

# Base project paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

CONFIG_PATH = PROJECT_ROOT / "config.yaml"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
APP_DIR = PROJECT_ROOT / "app"


def load_config(config_file: Path | str = CONFIG_PATH) -> Dict[str, Any]:
    """
    Load project configuration from a YAML file.

    Parameters
    ----------
    config_file : Path or str, default=CONFIG_PATH
        Path to the configuration YAML file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    config_path = Path(config_file)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}
