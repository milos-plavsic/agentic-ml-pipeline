from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA_SOURCE = (
    "UCI Machine Learning Repository — Student Performance (secondary school, Portugal). "
    "https://archive.ics.uci.edu/dataset/320/student+performance "
    "(Cortez & Silva, 2008)."
)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


_UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/"
_UCI_BASE_HTTP = "http://archive.ics.uci.edu/ml/machine-learning-databases/00320/"


def _download_first_working(urls: tuple[str, ...], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: BaseException | None = None
    for url in urls:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; portfolio-report/1.0)"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                dest.write_bytes(resp.read())
            return
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not download {dest.name!r}") from last_err


def _ensure_csv(name: str) -> Path:
    path = project_root() / "data" / name
    if path.exists():
        return path
    _download_first_working(
        (_UCI_BASE + name, _UCI_BASE_HTTP + name),
        path,
    )
    return path


def load_student_math() -> pd.DataFrame:
    path = _ensure_csv("student-mat.csv")
    return pd.read_csv(path, sep=";")


def prepare_regression_xy(df: pd.DataFrame, target: str = "G3") -> tuple[pd.DataFrame, np.ndarray]:
    y = df[target].to_numpy(dtype=np.float64)
    X = df.drop(columns=[target])
    return X, y
