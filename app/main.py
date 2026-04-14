import os

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from app.datasets import DATA_SOURCE, load_student_math, prepare_regression_xy


def run_pipeline(dataset_name: str = "uci_student_math") -> dict:
    df = load_student_math()
    X, y = prepare_regression_xy(df)
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=12,
        random_state=42,
        n_jobs=4,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        "dataset": dataset_name,
        "task": "regression",
        "target": "G3_final_math_grade",
        "n_rows": int(len(df)),
        "n_features_encoded": int(X.shape[1]),
        "model": "RandomForestRegressor",
        "test_mae": float(mean_absolute_error(y_test, pred)),
        "test_r2": float(r2_score(y_test, pred)),
        "data_source": DATA_SOURCE,
    }


def main() -> None:
    dataset_name = os.getenv("DEMO_DATASET", "uci_student_math")
    result = run_pipeline(dataset_name)
    print("Agentic ML Pipeline (real educational data)")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
