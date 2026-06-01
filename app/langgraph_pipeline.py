from __future__ import annotations

from typing import Any, Literal, TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, StateGraph
from ml_core import configure_logging
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from app.datasets import DATA_SOURCE, load_student_math
from app.orchestration_policy import (
    confidence_label,
    decide_loop,
    normalize_threshold,
    normalized_mae_quality,
    normalized_r2_quality,
    normalized_stability,
    weighted_confidence,
)

logger = configure_logging("langgraph_pipeline")


class PipelineState(TypedDict, total=False):
    """LangGraph pipeline state — all fields are optional so partial updates work."""

    dataset_name: str
    data: Any | None
    features: Any | None
    model: Any | None
    metrics: dict | None
    iteration: int
    confidence: float
    max_iterations: int
    confidence_threshold: float
    error: str | None

    # Extended internal fields
    confidence_label: str
    continue_loop: bool
    stop_reason: str

    raw_df: pd.DataFrame
    X_full: pd.DataFrame
    y_full: np.ndarray
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    y_pred: np.ndarray
    test_mae: float
    test_r2: float
    cv_mae_mean: float
    cv_mae_std: float
    cv_r2_mean: float
    cv_r2_std: float

    include_prior_grades: bool
    n_estimators: int
    max_depth: int | None
    min_samples_leaf: int

    model_name: str
    history: list[dict]
    decisions: list[str]

    # For classification mode (roc-auc / precision / recall)
    roc_auc: float | None
    precision: float | None
    recall: float | None
    task_type: str  # "regression" or "classification"


SUPPORTED_DATASET = "uci_student_math"


# ---------------------------------------------------------------------------
# Node: load_data_node
# ---------------------------------------------------------------------------


def load_data_node(state: PipelineState) -> PipelineState:
    """Load UCI student math dataset and store in state."""
    dataset_name = state.get("dataset_name", SUPPORTED_DATASET)
    if dataset_name != SUPPORTED_DATASET:
        return {"error": f"unsupported dataset: {dataset_name!r}"}

    # Initialise loop bookkeeping on first call
    updates: PipelineState = {}
    if state.get("confidence_threshold") is None:
        updates["confidence_threshold"] = normalize_threshold(0.68)
    if state.get("max_iterations") is None:
        updates["max_iterations"] = 3
    if state.get("iteration") is None:
        updates["iteration"] = 0
    if state.get("history") is None:
        updates["history"] = []
    if state.get("decisions") is None:
        updates["decisions"] = []
    if state.get("model_name") is None:
        updates["model_name"] = "RandomForestRegressor"

    df = load_student_math()
    y = df["G3"].to_numpy(dtype=np.float64)
    logger.info(f"Loaded dataset: {df.shape[0]} rows")
    updates["data"] = df
    updates["raw_df"] = df
    updates["y_full"] = y
    updates["dataset_name"] = dataset_name
    updates["task_type"] = "regression"
    updates["error"] = None
    return updates


# ---------------------------------------------------------------------------
# Node: preprocess_node
# ---------------------------------------------------------------------------


def preprocess_node(state: PipelineState) -> PipelineState:
    """Feature engineering: encode categoricals, handle nulls, train/test split."""
    df: pd.DataFrame = state["raw_df"]
    it = int(state.get("iteration", 0)) + 1
    include_prior_grades = it >= 2

    cols_to_drop = ["G3"]
    if not include_prior_grades:
        cols_to_drop.extend(["G1", "G2"])

    Xdf = df.drop(columns=cols_to_drop)
    # One-hot encode all categorical columns
    Xdf = pd.get_dummies(Xdf, drop_first=True)

    # Handle any remaining nulls
    Xdf = Xdf.fillna(Xdf.median(numeric_only=True))

    n_estimators = 24 if it == 1 else 48
    max_depth: int | None = 10 if it == 1 else 12
    min_samples_leaf = 2 if it == 1 else 1

    decision = (
        f"iteration={it}: include_prior_grades={include_prior_grades}, "
        f"n_estimators={n_estimators}, max_depth={max_depth}"
    )

    rng = int(state.get("random_state", 42)) if hasattr(state, "get") else 42  # type: ignore[call-overload]
    X_train, X_test, y_train, y_test = train_test_split(
        Xdf,
        state["y_full"],
        test_size=0.2,
        random_state=rng,
    )

    return {
        "features": Xdf,
        "X_full": Xdf,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "iteration": it,
        "include_prior_grades": include_prior_grades,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "decisions": [*state.get("decisions", []), decision],
    }


# ---------------------------------------------------------------------------
# Node: train_node
# ---------------------------------------------------------------------------


def train_node(state: PipelineState) -> PipelineState:
    """Train RandomForest / GradientBoosting model and compute CV metrics."""
    it = int(state.get("iteration", 1))

    # Alternate between RF and GB on successive iterations to explore
    if it % 2 == 0:
        model = GradientBoostingClassifier(
            n_estimators=state.get("n_estimators", 48),
            max_depth=state.get("max_depth", 5) or 5,
            random_state=42,
        )
        model_name = "GradientBoostingClassifier"
    else:
        model = RandomForestRegressor(
            n_estimators=state.get("n_estimators", 24),
            max_depth=state.get("max_depth", 10),
            min_samples_leaf=state.get("min_samples_leaf", 2),
            random_state=42,
            n_jobs=1,
        )
        model_name = "RandomForestRegressor"

    X_train = state["X_train"]
    y_train = state["y_train"]

    model.fit(X_train, y_train)
    pred = model.predict(state["X_test"])

    mae = float(mean_absolute_error(state["y_test"], pred))
    r2 = float(r2_score(state["y_test"], pred))

    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_mae = -cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )
    cv_r2 = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="r2",
        n_jobs=1,
    )

    return {
        "model": model,
        "model_name": model_name,
        "y_pred": pred,
        "test_mae": mae,
        "test_r2": r2,
        "cv_mae_mean": float(np.mean(cv_mae)),
        "cv_mae_std": float(np.std(cv_mae)),
        "cv_r2_mean": float(np.mean(cv_r2)),
        "cv_r2_std": float(np.std(cv_r2)),
    }


# ---------------------------------------------------------------------------
# Node: evaluate_node
# ---------------------------------------------------------------------------


def evaluate_node(state: PipelineState) -> PipelineState:
    """Compute confidence score from metrics; decide whether to re-train."""
    components = {
        "primary_quality": normalized_mae_quality(state["test_mae"]),
        "secondary_quality": normalized_r2_quality(state["test_r2"]),
        "stability": normalized_stability(state["cv_r2_std"]),
    }
    score = float(weighted_confidence(components))
    label = confidence_label(score)

    threshold = float(state.get("confidence_threshold", 0.68))
    it = int(state.get("iteration", 1))
    max_it = int(state.get("max_iterations", 3))

    loop = decide_loop(
        confidence_score=score,
        confidence_threshold=threshold,
        iteration=it,
        max_iterations=max_it,
    )

    iteration_summary = {
        "iteration": it,
        "include_prior_grades": state.get("include_prior_grades", False),
        "n_estimators": state.get("n_estimators", 24),
        "max_depth": state.get("max_depth"),
        "min_samples_leaf": state.get("min_samples_leaf", 2),
        "n_features_encoded": int(state["X_full"].shape[1]),
        "test_mae": state["test_mae"],
        "test_r2": state["test_r2"],
        "cv_mae_mean": state["cv_mae_mean"],
        "cv_mae_std": state["cv_mae_std"],
        "cv_r2_mean": state["cv_r2_mean"],
        "cv_r2_std": state["cv_r2_std"],
        "confidence_score": score,
    }

    # Also compute ROC-AUC / precision / recall for binary classification proxy
    # (G3 >= 10 is considered a pass)
    y_test = state["y_test"]
    y_pred_raw = state["y_pred"]
    y_binary = (y_test >= 10).astype(int)
    y_pred_binary = (y_pred_raw >= 10).astype(int)

    roc_auc: float | None = None
    precision: float | None = None
    recall: float | None = None

    try:
        if len(np.unique(y_binary)) > 1:
            roc_auc = float(roc_auc_score(y_binary, y_pred_binary))
            precision = float(precision_score(y_binary, y_pred_binary, zero_division=0))
            recall = float(recall_score(y_binary, y_pred_binary, zero_division=0))
    except Exception:
        pass

    return {
        "metrics": {
            "test_mae": state["test_mae"],
            "test_r2": state["test_r2"],
            "confidence_score": score,
            "roc_auc": roc_auc,
            "precision": precision,
            "recall": recall,
        },
        "confidence": score,
        "confidence_label": label,
        "continue_loop": loop["continue_loop"],
        "stop_reason": loop["stop_reason"],
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "history": [*state.get("history", []), iteration_summary],
    }


# ---------------------------------------------------------------------------
# Node: report_node
# ---------------------------------------------------------------------------


def report_node(state: PipelineState) -> PipelineState:
    """Format the final report dictionary and store it in state."""
    report = {
        "dataset": state.get("dataset_name", SUPPORTED_DATASET),
        "task": "regression",
        "target": "G3_final_math_grade",
        "n_rows": int(len(state["raw_df"])),
        "n_features_encoded": int(state["X_full"].shape[1]),
        "model": state.get("model_name", "RandomForestRegressor"),
        "test_mae": float(state["test_mae"]),
        "test_r2": float(state["test_r2"]),
        "cv_mae_mean": float(state["cv_mae_mean"]),
        "cv_mae_std": float(state["cv_mae_std"]),
        "cv_r2_mean": float(state["cv_r2_mean"]),
        "cv_r2_std": float(state["cv_r2_std"]),
        "confidence_score": float(state.get("confidence", state.get("confidence_score", 0.0))),
        "confidence_label": state.get("confidence_label", "low"),
        "confidence_threshold": float(state.get("confidence_threshold", 0.68)),
        "iterations": int(state.get("iteration", 1)),
        "loop_terminated_reason": state.get("stop_reason", "max_iterations_reached"),
        "iteration_history": state.get("history", []),
        "decision_log": state.get("decisions", []),
        "data_source": DATA_SOURCE,
        "roc_auc": state.get("roc_auc"),
        "precision": state.get("precision"),
        "recall": state.get("recall"),
    }
    return {"metrics": report}


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------


def _route_after_evaluate(state: PipelineState) -> Literal["preprocess", "report"]:
    """Route back to preprocess for another iteration, or finalize."""
    return "preprocess" if state.get("continue_loop", False) else "report"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_pipeline_graph() -> Any:
    """Build and compile the LangGraph pipeline."""
    graph = StateGraph(PipelineState)

    graph.add_node("load", load_data_node)
    graph.add_node("preprocess", preprocess_node)
    graph.add_node("train", train_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("load")
    graph.add_edge("load", "preprocess")
    graph.add_edge("preprocess", "train")
    graph.add_edge("train", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        _route_after_evaluate,
        {
            "preprocess": "preprocess",
            "report": "report",
        },
    )
    graph.add_edge("report", END)

    return graph.compile()


_PIPELINE_GRAPH = build_pipeline_graph()


def run_agentic_pipeline(
    dataset_name: str = SUPPORTED_DATASET,
    *,
    confidence_threshold: float = 0.68,
    max_iterations: int = 3,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run the compiled LangGraph pipeline and return a serialisable result dict."""
    final_state = _PIPELINE_GRAPH.invoke(
        {
            "dataset_name": dataset_name,
            "confidence_threshold": normalize_threshold(confidence_threshold),
            "max_iterations": max(1, int(max_iterations)),
            "random_state": int(random_state),
            "iteration": 0,
            "history": [],
            "decisions": [],
            "model_name": "RandomForestRegressor",
        }
    )

    # The report node puts the finished report in state["metrics"]
    report = final_state.get("metrics") or {}
    if not report:
        # Fallback: build from state directly
        report = {
            "dataset": final_state.get("dataset_name", dataset_name),
            "task": "regression",
            "target": "G3_final_math_grade",
            "n_rows": int(len(final_state.get("raw_df", []))),
            "n_features_encoded": int(final_state.get("X_full", pd.DataFrame()).shape[1]),
            "model": final_state.get("model_name", "RandomForestRegressor"),
            "test_mae": float(final_state.get("test_mae", 0.0)),
            "test_r2": float(final_state.get("test_r2", 0.0)),
            "cv_mae_mean": float(final_state.get("cv_mae_mean", 0.0)),
            "cv_mae_std": float(final_state.get("cv_mae_std", 0.0)),
            "cv_r2_mean": float(final_state.get("cv_r2_mean", 0.0)),
            "cv_r2_std": float(final_state.get("cv_r2_std", 0.0)),
            "confidence_score": float(final_state.get("confidence", 0.0)),
            "confidence_label": final_state.get("confidence_label", "low"),
            "confidence_threshold": float(
                final_state.get("confidence_threshold", confidence_threshold)
            ),
            "iterations": int(final_state.get("iteration", 1)),
            "loop_terminated_reason": final_state.get("stop_reason", "max_iterations_reached"),
            "iteration_history": final_state.get("history", []),
            "decision_log": final_state.get("decisions", []),
            "data_source": DATA_SOURCE,
        }

    return report
