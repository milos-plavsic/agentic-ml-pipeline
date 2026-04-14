from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from app.datasets import DATA_SOURCE, load_student_math, prepare_regression_xy


def run_rf_hyperparam_finetune(random_state: int = 42) -> dict:
    """Random search over RandomForest hyperparameters (tabular “fine-tuning” of the inductive bias)."""
    n_iter = int(os.getenv("FINETUNE_N_ITER", "14"))
    df = load_student_math()
    X, y = prepare_regression_xy(df)
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    param = {
        "n_estimators": randint(80, 400),
        "max_depth": [None, 8, 10, 12, 16],
        "min_samples_leaf": randint(1, 6),
        "max_features": ["sqrt", "log2", None],
    }
    base = RandomForestRegressor(random_state=random_state, n_jobs=2)
    search = RandomizedSearchCV(
        base,
        param,
        n_iter=n_iter,
        cv=3,
        scoring="neg_mean_absolute_error",
        random_state=random_state,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    pred = search.predict(X_test)
    best = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in search.best_params_.items()}
    return {
        "best_params": best,
        "test_mae": float(mean_absolute_error(y_test, pred)),
        "test_r2": float(r2_score(y_test, pred)),
        "n_iter": n_iter,
        "data_source": DATA_SOURCE,
    }


def main() -> None:
    out = run_rf_hyperparam_finetune()
    print("RandomForest hyperparameter search (fine-tune)")
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
