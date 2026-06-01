from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.studio.datasets import load_dataframe


def _prepare_xy(df: pd.DataFrame, target: str, task: str) -> tuple[pd.DataFrame, np.ndarray]:
    y = df[target]
    X = df.drop(columns=[target])
    if task == "classification":
        numeric = pd.to_numeric(y, errors="coerce")
        if numeric.notna().sum() > len(y) * 0.5:
            y_arr = (numeric >= 10).astype(int).to_numpy()
        else:
            y_arr = (
                y.astype(str).str.lower().isin(["pass", "yes", "true", "1"]).astype(int).to_numpy()
            )
    else:
        y_arr = pd.to_numeric(y, errors="coerce").to_numpy()
    return X, y_arr


def run_ensemble_benchmark(
    dataset_id: str,
    *,
    cv_splits: int = 3,
) -> dict[str, Any]:
    """RF / XGB / CatBoost-style comparison (RF always; boosters if installed)."""
    from app.studio.datasets import get_dataset

    meta = get_dataset(dataset_id)
    df = load_dataframe(dataset_id)
    X, y = _prepare_xy(df, meta["target_column"], meta["task"])

    cat_cols = [c for c in X.columns if X[c].dtype == object or X[c].nunique() < 20]
    num_cols = [c for c in X.columns if c not in cat_cols]
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    task = meta["task"]
    scoring = "roc_auc" if task == "classification" else "neg_mean_absolute_error"
    models: dict[str, object] = {
        "random_forest": (
            RandomForestClassifier(n_estimators=120, random_state=42)
            if task == "classification"
            else RandomForestRegressor(n_estimators=120, random_state=42)
        ),
    }
    try:
        from xgboost import XGBClassifier, XGBRegressor

        models["xgboost"] = (
            XGBClassifier(n_estimators=80, max_depth=4, eval_metric="logloss", random_state=42)
            if task == "classification"
            else XGBRegressor(n_estimators=80, max_depth=4, random_state=42)
        )
    except ImportError:
        pass
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor

        models["catboost"] = (
            CatBoostClassifier(iterations=80, depth=4, verbose=False, random_state=42)
            if task == "classification"
            else CatBoostRegressor(iterations=80, depth=4, verbose=False, random_state=42)
        )
    except ImportError:
        pass

    leaderboard = []
    for name, estimator in models.items():
        pipe = Pipeline([("pre", pre), ("model", estimator)])
        scores = cross_val_score(pipe, X, y, cv=cv_splits, scoring=scoring, n_jobs=1)
        leaderboard.append(
            {
                "model": name,
                "metric": scoring,
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
            }
        )
    leaderboard.sort(key=lambda r: r["mean"], reverse=scoring.startswith("roc"))
    return {
        "dataset_id": dataset_id,
        "task": task,
        "target_column": meta["target_column"],
        "leaderboard": leaderboard,
        "winner": leaderboard[0]["model"] if leaderboard else None,
    }
