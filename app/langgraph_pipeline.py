from __future__ import annotations

from typing import Any, Literal, TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, StateGraph
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from app.datasets import DATA_SOURCE, load_student_math
from app.orchestration_policy import (
    confidence_label,
    decide_loop,
    normalized_mae_quality,
    normalized_r2_quality,
    normalized_stability,
    normalize_threshold,
    weighted_confidence,
)


class IterationMetrics(TypedDict):
    iteration: int
    include_prior_grades: bool
    n_estimators: int
    max_depth: int | None
    min_samples_leaf: int
    n_features_encoded: int
    test_mae: float
    test_r2: float
    cv_mae_mean: float
    cv_mae_std: float
    cv_r2_mean: float
    cv_r2_std: float
    confidence_score: float


class PipelineState(TypedDict, total=False):
    dataset_name: str
    confidence_threshold: float
    max_iterations: int
    random_state: int

    iteration: int
    include_prior_grades: bool
    n_estimators: int
    max_depth: int | None
    min_samples_leaf: int

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

    confidence_score: float
    confidence_label: str
    continue_loop: bool
    stop_reason: str

    model_name: str
    history: list[IterationMetrics]
    decisions: list[str]


SUPPORTED_DATASET = "uci_student_math"


def _validate_request(state: PipelineState) -> PipelineState:
    ds = state.get("dataset_name", SUPPORTED_DATASET)
    if ds != SUPPORTED_DATASET:
        raise ValueError(f"unsupported dataset: {ds!r}")

    return {
        "dataset_name": ds,
        "confidence_threshold": normalize_threshold(state.get("confidence_threshold", 0.68)),
        "max_iterations": max(1, int(state.get("max_iterations", 3))),
        "random_state": int(state.get("random_state", 42)),
        "iteration": 0,
        "history": [],
        "decisions": [],
        "model_name": "RandomForestRegressor",
    }


def _load_dataset(state: PipelineState) -> PipelineState:
    df = load_student_math()
    y = df["G3"].to_numpy(dtype=np.float64)
    return {"raw_df": df, "y_full": y}


def _plan_iteration(state: PipelineState) -> PipelineState:
    it = int(state["iteration"]) + 1
    include_prior = it >= 2
    n_estimators = 260 if it == 1 else 420
    max_depth: int | None = 12 if it == 1 else None
    min_samples_leaf = 2 if it == 1 else 1

    decision = (
        f"iteration={it}: include_prior_grades={include_prior}, "
        f"n_estimators={n_estimators}, max_depth={max_depth}, min_samples_leaf={min_samples_leaf}"
    )

    return {
        "iteration": it,
        "include_prior_grades": include_prior,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "decisions": [*state["decisions"], decision],
    }


def _prepare_features(state: PipelineState) -> PipelineState:
    df = state["raw_df"]
    cols_to_drop = ["G3"]
    if not state["include_prior_grades"]:
        cols_to_drop.extend(["G1", "G2"])

    Xdf = df.drop(columns=cols_to_drop)
    Xdf = pd.get_dummies(Xdf, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        Xdf,
        state["y_full"],
        test_size=0.2,
        random_state=state["random_state"],
    )

    return {
        "X_full": Xdf,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def _train_and_evaluate(state: PipelineState) -> PipelineState:
    model = RandomForestRegressor(
        n_estimators=state["n_estimators"],
        max_depth=state["max_depth"],
        min_samples_leaf=state["min_samples_leaf"],
        random_state=state["random_state"],
        n_jobs=4,
    )

    model.fit(state["X_train"], state["y_train"])
    pred = model.predict(state["X_test"])
    mae = float(mean_absolute_error(state["y_test"], pred))
    r2 = float(r2_score(state["y_test"], pred))

    cv = KFold(n_splits=3, shuffle=True, random_state=state["random_state"])
    cv_mae = -cross_val_score(
        model,
        state["X_train"],
        state["y_train"],
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )
    cv_r2 = cross_val_score(
        model,
        state["X_train"],
        state["y_train"],
        cv=cv,
        scoring="r2",
        n_jobs=1,
    )

    return {
        "y_pred": pred,
        "test_mae": mae,
        "test_r2": r2,
        "cv_mae_mean": float(np.mean(cv_mae)),
        "cv_mae_std": float(np.std(cv_mae)),
        "cv_r2_mean": float(np.mean(cv_r2)),
        "cv_r2_std": float(np.std(cv_r2)),
    }


def _assess_confidence(state: PipelineState) -> PipelineState:
    components = {
        "primary_quality": normalized_mae_quality(state["test_mae"]),
        "secondary_quality": normalized_r2_quality(state["test_r2"]),
        "stability": normalized_stability(state["cv_r2_std"]),
    }
    score = weighted_confidence(components)
    label = confidence_label(score)

    loop = decide_loop(
        confidence_score=score,
        confidence_threshold=state["confidence_threshold"],
        iteration=state["iteration"],
        max_iterations=state["max_iterations"],
    )

    iteration_summary: IterationMetrics = {
        "iteration": state["iteration"],
        "include_prior_grades": state["include_prior_grades"],
        "n_estimators": state["n_estimators"],
        "max_depth": state["max_depth"],
        "min_samples_leaf": state["min_samples_leaf"],
        "n_features_encoded": int(state["X_full"].shape[1]),
        "test_mae": state["test_mae"],
        "test_r2": state["test_r2"],
        "cv_mae_mean": state["cv_mae_mean"],
        "cv_mae_std": state["cv_mae_std"],
        "cv_r2_mean": state["cv_r2_mean"],
        "cv_r2_std": state["cv_r2_std"],
        "confidence_score": score,
    }

    return {
        "confidence_score": score,
        "confidence_label": label,
        "continue_loop": loop["continue_loop"],
        "stop_reason": loop["stop_reason"],
        "history": [*state["history"], iteration_summary],
    }


def _route_after_assessment(state: PipelineState) -> Literal["prepare_features", "finalize"]:
    return "prepare_features" if state["continue_loop"] else "finalize"


def _finalize(state: PipelineState) -> PipelineState:
    return {"stop_reason": state["stop_reason"]}


def build_pipeline_graph():
    g = StateGraph(PipelineState)

    g.add_node("validate_request", _validate_request)
    g.add_node("load_dataset", _load_dataset)
    g.add_node("plan_iteration", _plan_iteration)
    g.add_node("prepare_features", _prepare_features)
    g.add_node("train_and_evaluate", _train_and_evaluate)
    g.add_node("assess_confidence", _assess_confidence)
    g.add_node("finalize", _finalize)

    g.set_entry_point("validate_request")
    g.add_edge("validate_request", "load_dataset")
    g.add_edge("load_dataset", "plan_iteration")
    g.add_edge("plan_iteration", "prepare_features")
    g.add_edge("prepare_features", "train_and_evaluate")
    g.add_edge("train_and_evaluate", "assess_confidence")
    g.add_conditional_edges(
        "assess_confidence",
        _route_after_assessment,
        {
            "prepare_features": "plan_iteration",
            "finalize": "finalize",
        },
    )
    g.add_edge("finalize", END)

    return g.compile()


_PIPELINE_GRAPH = build_pipeline_graph()


def run_agentic_pipeline(
    dataset_name: str = SUPPORTED_DATASET,
    *,
    confidence_threshold: float = 0.68,
    max_iterations: int = 3,
    random_state: int = 42,
) -> dict[str, Any]:
    final_state = _PIPELINE_GRAPH.invoke(
        {
            "dataset_name": dataset_name,
            "confidence_threshold": confidence_threshold,
            "max_iterations": max_iterations,
            "random_state": random_state,
        }
    )

    return {
        "dataset": final_state["dataset_name"],
        "task": "regression",
        "target": "G3_final_math_grade",
        "n_rows": int(len(final_state["raw_df"])),
        "n_features_encoded": int(final_state["X_full"].shape[1]),
        "model": final_state["model_name"],
        "test_mae": float(final_state["test_mae"]),
        "test_r2": float(final_state["test_r2"]),
        "cv_mae_mean": float(final_state["cv_mae_mean"]),
        "cv_mae_std": float(final_state["cv_mae_std"]),
        "cv_r2_mean": float(final_state["cv_r2_mean"]),
        "cv_r2_std": float(final_state["cv_r2_std"]),
        "confidence_score": float(final_state["confidence_score"]),
        "confidence_label": final_state["confidence_label"],
        "confidence_threshold": float(final_state["confidence_threshold"]),
        "iterations": int(final_state["iteration"]),
        "loop_terminated_reason": final_state["stop_reason"],
        "iteration_history": final_state["history"],
        "decision_log": final_state["decisions"],
        "data_source": DATA_SOURCE,
    }
