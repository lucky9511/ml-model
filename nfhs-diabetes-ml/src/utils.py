"""
Utility functions for nfhs-diabetes-ml project.
Includes seed initialization, logging setup, and artifact persistence helpers.
"""

import logging
import os
import random
from pathlib import Path
from typing import Any, Optional
import joblib
import numpy as np

from src.config import RANDOM_STATE


def set_seed(seed: int = RANDOM_STATE) -> None:
    """
    Set seeds across Python standard library and NumPy for reproducibility.

    Parameters
    ----------
    seed : int, default=RANDOM_STATE (42)
        Random seed value to enforce deterministic operations.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logger(
    name: str = "nfhs-diabetes-ml",
    level: int = logging.INFO,
    log_file: Optional[Path | str] = None,
) -> logging.Logger:
    """
    Configure and return a standard formatted logger.

    Parameters
    ----------
    name : str
        Logger name.
    level : int, default=logging.INFO
        Logging severity level.
    log_file : Path or str, optional
        Filepath to write logs to.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if logger was already created
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def ensure_directory(path: Path | str) -> Path:
    """
    Ensure that a directory exists, creating it if necessary.

    Parameters
    ----------
    path : Path or str
        Directory path to verify or create.

    Returns
    -------
    Path
        Resolved Path object of the ensured directory.
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_artifact(obj: Any, file_path: Path | str) -> Path:
    """
    Save a Python object/model artifact using joblib.

    Parameters
    ----------
    obj : Any
        Model or object to persist.
    file_path : Path or str
        Destination path.

    Returns
    -------
    Path
        Destination path where artifact was saved.
    """
    dest = Path(file_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, dest)
    return dest


def load_artifact(file_path: Path | str) -> Any:
    """
    Load a persisted Python object/model artifact using joblib.

    Parameters
    ----------
    file_path : Path or str
        Path to the saved artifact.

    Returns
    -------
    Any
        Loaded Python object.
    """
    source = Path(file_path)
    if not source.is_file():
        raise FileNotFoundError(f"Artifact not found at: {source}")
    return joblib.load(source)
