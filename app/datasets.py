"""Data loading with validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from ml_core import configure_logging, validate_dataframe

from app.uci_fetch import fetch_uci_student_csv

logger = configure_logging("datasets")

DATA_SOURCE = (
    "UCI Machine Learning Repository — Student Performance (secondary school, Portugal). "
    "https://archive.ics.uci.edu/dataset/320/student+performance "
    "(Cortez & Silva, 2008)."
)

# Define explicit dtypes for performance and consistency
STUDENT_DTYPES = {
    "school": "category",
    "sex": "category",
    "age": "int8",
    "address": "category",
    "famsize": "category",
    "Pstatus": "category",
    "Medu": "int8",
    "Fedu": "int8",
    "Mjob": "category",
    "Fjob": "category",
    "reason": "category",
    "guardian": "category",
    "traveltime": "int8",
    "studytime": "int8",
    "failures": "int8",
    "schoolsup": "category",
    "famsup": "category",
    "paid": "category",
    "activities": "category",
    "nursery": "category",
    "higher": "category",
    "internet": "category",
    "romantic": "category",
    "famrel": "int8",
    "freetime": "int8",
    "goout": "int8",
    "Dalc": "int8",
    "Walc": "int8",
    "health": "int8",
    "absences": "int8",
    "G1": "float32",
    "G2": "float32",
    "G3": "float32",
}


def project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).resolve().parent.parent


def _ensure_csv(name: str) -> Path:
    """Ensure CSV file exists, download if needed."""
    path = project_root() / "data" / name
    if path.exists():
        logger.info(f"Found local data file: {path}")
        return path

    try:
        logger.info(f"Downloading {name}...")
        fetch_uci_student_csv(name, path)
        logger.info(f"Downloaded to {path}")
    except Exception as e:
        raise RuntimeError(f"Could not obtain UCI file {name!r}: {e}") from e

    return path


def load_student_math() -> pd.DataFrame:
    """Load student math performance dataset.

    Returns:
        DataFrame with explicit dtypes and validation

    Raises:
        RuntimeError: If data cannot be loaded or is invalid
    """
    path = _ensure_csv("student-mat.csv")

    try:
        df = pd.read_csv(
            path,
            sep=";",
            dtype=STUDENT_DTYPES,
            na_values=["NA", "N/A", ""],
        )
        logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV {path}: {e}") from e

    # Validate using ml_core (raises ValidationError if nulls or empty)
    validate_dataframe(df)

    return df


def prepare_regression_xy(
    df: pd.DataFrame,
    target: str = "G3",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Prepare features and target for regression.

    Args:
        df: Input DataFrame
        target: Target column name

    Returns:
        Tuple of (features, target)

    Raises:
        ValueError: If validation fails
    """
    # Validate input
    validate_dataframe(df)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")

    # Convert categorical to numeric
    X = df.drop(columns=[target]).copy()
    for col in X.select_dtypes(include=["category"]).columns:
        X[col] = X[col].cat.codes.astype("int8")

    # Extract target
    y = df[target].astype("float32").values

    logger.info(f"Prepared data: X shape {X.shape}, y shape {y.shape}")

    return X, y
