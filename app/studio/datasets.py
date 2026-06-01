from __future__ import annotations

import json
import uuid
from pathlib import Path

import pandas as pd
from ml_core import configure_logging

logger = configure_logging("studio.datasets")

UPLOAD_ROOT = Path(__file__).resolve().parents[2] / "data" / "uploads"
REGISTRY_PATH = UPLOAD_ROOT / "registry.json"


def _load_registry() -> dict[str, dict]:
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_PATH.exists():
        return {}
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _save_registry(reg: dict[str, dict]) -> None:
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2), encoding="utf-8")


def register_upload(filename: str, content: bytes) -> dict:
    """Persist an uploaded CSV and infer schema metadata."""
    dataset_id = uuid.uuid4().hex[:12]
    safe_name = Path(filename).name.replace(" ", "_")
    path = UPLOAD_ROOT / f"{dataset_id}_{safe_name}"
    path.write_bytes(content)

    sep = ";" if safe_name.endswith("-mat.csv") or "student" in safe_name.lower() else ","
    df = pd.read_csv(path, sep=sep, nrows=5000)
    target = "G3" if "G3" in df.columns else df.columns[-1]
    task = "classification" if df[target].nunique() <= 10 else "regression"

    meta = {
        "dataset_id": dataset_id,
        "filename": safe_name,
        "path": str(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "target_column": target,
        "task": task,
        "separator": sep,
    }
    reg = _load_registry()
    reg[dataset_id] = meta
    _save_registry(reg)
    logger.info("registered dataset %s (%s rows)", dataset_id, meta["rows"])
    return meta


def list_datasets() -> list[dict]:
    return list(_load_registry().values())


def get_dataset(dataset_id: str) -> dict:
    reg = _load_registry()
    if dataset_id not in reg:
        raise KeyError(f"unknown dataset_id: {dataset_id}")
    return reg[dataset_id]


def load_dataframe(dataset_id: str) -> pd.DataFrame:
    meta = get_dataset(dataset_id)
    return pd.read_csv(meta["path"], sep=meta.get("separator", ","))
