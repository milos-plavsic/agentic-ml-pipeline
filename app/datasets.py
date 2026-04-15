from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.uci_fetch import fetch_uci_student_csv

DATA_SOURCE = (
    "UCI Machine Learning Repository — Student Performance (secondary school, Portugal). "
    "https://archive.ics.uci.edu/dataset/320/student+performance "
    "(Cortez & Silva, 2008)."
)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ensure_csv(name: str) -> Path:
    path = project_root() / "data" / name
    if path.exists():
        return path
    try:
        fetch_uci_student_csv(name, path)
    except Exception as e:
        raise RuntimeError(f"Could not obtain UCI file {name!r}") from e
    return path


def load_student_math() -> pd.DataFrame:
    path = _ensure_csv("student-mat.csv")
    return pd.read_csv(path, sep=";")


def prepare_regression_xy(df: pd.DataFrame, target: str = "G3") -> tuple[pd.DataFrame, np.ndarray]:
    y = df[target].to_numpy(dtype=np.float64)
    X = df.drop(columns=[target])
    return X, y
