from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from analysis.json_util import dumps_pretty
from analysis.plotting import residual_histogram, scatter_actual_vs_predicted
from analysis.stats_utils import regression_summary
from app.datasets import DATA_SOURCE, load_student_math, prepare_regression_xy


def generate_report(out_dir: Path | None = None, random_state: int = 42) -> dict:
    out = Path(out_dir or "reports")
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)

    df = load_student_math()
    X, y = prepare_regression_xy(df)
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    model = RandomForestRegressor(
        n_estimators=250, max_depth=12, random_state=random_state, n_jobs=4
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    stats = regression_summary(y_test, y_pred)
    summary = {
        "data_source": DATA_SOURCE,
        "target": "G3",
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features_encoded": int(X.shape[1]),
        "metrics_test": stats,
    }
    (out / "summary.json").write_text(dumps_pretty(summary), encoding="utf-8")

    scatter_actual_vs_predicted(
        np.asarray(y_test), y_pred, fig_dir / "actual_vs_predicted.png"
    )
    residual_histogram(np.asarray(y_test), y_pred, fig_dir / "residuals.png")

    def _fmt(x: float) -> str:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return "n/a"
        return f"{x:.3f}"

    md = "\n".join(
        [
            "# Regression report — UCI student math (G3)",
            "",
            f"**Data:** {DATA_SOURCE}",
            "",
            "## Test-set metrics",
            "",
            f"- MAE: **{_fmt(stats['mae'])}**",
            f"- RMSE: **{_fmt(stats['rmse'])}**",
            f"- R²: **{_fmt(stats['r2'])}**",
            f"- Residual mean ± std: **{_fmt(stats['residual_mean'])}** ± **{_fmt(stats['residual_std'])}**",
            "",
            "## Figures",
            "",
            "- `figures/actual_vs_predicted.png`",
            "- `figures/residuals.png`",
        ]
    )
    (out / "REPORT.md").write_text(md, encoding="utf-8")
    return {"output_dir": str(out.resolve())}


def main() -> None:
    print(dumps_pretty(generate_report()))


if __name__ == "__main__":
    main()
