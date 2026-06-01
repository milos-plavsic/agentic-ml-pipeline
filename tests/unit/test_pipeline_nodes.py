"""Unit tests for individual LangGraph pipeline nodes.

Each test exercises a node function in isolation using a synthetic or real
minimal state dict — no network access, no heavy graph compilation needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers: build minimal states
# ---------------------------------------------------------------------------

_BASE_STATE: dict = {
    "dataset_name": "uci_student_math",
    "confidence_threshold": 0.68,
    "max_iterations": 3,
    "iteration": 0,
    "history": [],
    "decisions": [],
    "model_name": "RandomForestRegressor",
    "random_state": 42,
    "error": None,
}


def _synthetic_df(n: int = 80) -> pd.DataFrame:
    """Return a minimal DataFrame shaped like the UCI student-math CSV."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "age": rng.integers(15, 22, size=n).astype("int8"),
            "Medu": rng.integers(0, 5, size=n).astype("int8"),
            "Fedu": rng.integers(0, 5, size=n).astype("int8"),
            "traveltime": rng.integers(1, 5, size=n).astype("int8"),
            "studytime": rng.integers(1, 5, size=n).astype("int8"),
            "failures": rng.integers(0, 4, size=n).astype("int8"),
            "famrel": rng.integers(1, 6, size=n).astype("int8"),
            "freetime": rng.integers(1, 6, size=n).astype("int8"),
            "goout": rng.integers(1, 6, size=n).astype("int8"),
            "Dalc": rng.integers(1, 6, size=n).astype("int8"),
            "Walc": rng.integers(1, 6, size=n).astype("int8"),
            "health": rng.integers(1, 6, size=n).astype("int8"),
            "absences": rng.integers(0, 20, size=n).astype("int8"),
            "sex": pd.Categorical(rng.choice(["M", "F"], size=n)),
            "address": pd.Categorical(rng.choice(["U", "R"], size=n)),
            "G1": rng.uniform(5, 18, size=n).astype("float32"),
            "G2": rng.uniform(5, 18, size=n).astype("float32"),
            "G3": rng.uniform(5, 18, size=n).astype("float32"),
        }
    )
    return df


def _state_with_data(n: int = 80) -> dict:
    df = _synthetic_df(n)
    y = df["G3"].to_numpy(dtype=np.float64)
    state = dict(_BASE_STATE)
    state.update(
        {
            "data": df,
            "raw_df": df,
            "y_full": y,
            "task_type": "regression",
        }
    )
    return state


def _state_after_preprocess(n: int = 80) -> dict:
    from app.langgraph_pipeline import preprocess_node

    state = _state_with_data(n)
    return {**state, **preprocess_node(state)}


def _state_after_train(n: int = 80) -> dict:
    from app.langgraph_pipeline import train_node

    state = _state_after_preprocess(n)
    return {**state, **train_node(state)}


# ---------------------------------------------------------------------------
# Tests: load_data_node
# ---------------------------------------------------------------------------


class TestLoadDataNode:
    """Tests for load_data_node."""

    def test_load_data_node_returns_non_null_data(self):
        """load_data_node must populate state['data'] with the real CSV."""
        from app.langgraph_pipeline import load_data_node

        state = {**_BASE_STATE}
        result = load_data_node(state)
        assert result.get("data") is not None, "state['data'] must not be None after load"

    def test_load_data_node_returns_dataframe(self):
        """state['data'] must be a pandas DataFrame."""
        from app.langgraph_pipeline import load_data_node

        result = load_data_node({**_BASE_STATE})
        assert isinstance(result["data"], pd.DataFrame)

    def test_load_data_node_has_sufficient_rows(self):
        """Dataset must have at least 300 rows (UCI student-mat has 395)."""
        from app.langgraph_pipeline import load_data_node

        result = load_data_node({**_BASE_STATE})
        assert len(result["data"]) >= 300

    def test_load_data_node_sets_y_full(self):
        """load_data_node must set y_full as a numpy array."""
        from app.langgraph_pipeline import load_data_node

        result = load_data_node({**_BASE_STATE})
        assert isinstance(result["y_full"], np.ndarray)

    def test_load_data_node_rejects_unsupported_dataset(self):
        """Unsupported dataset name must set state['error'], not raise."""
        from app.langgraph_pipeline import load_data_node

        state = {**_BASE_STATE, "dataset_name": "not_a_real_dataset"}
        result = load_data_node(state)
        assert result.get("error") is not None


# ---------------------------------------------------------------------------
# Tests: preprocess_node
# ---------------------------------------------------------------------------


class TestPreprocessNode:
    """Tests for preprocess_node."""

    def test_preprocess_node_returns_features(self):
        """preprocess_node must return an X_full DataFrame."""
        from app.langgraph_pipeline import preprocess_node

        state = _state_with_data()
        result = preprocess_node(state)
        assert result.get("X_full") is not None

    def test_preprocess_node_encodes_categoricals(self):
        """All columns in X_full must be numeric after one-hot encoding."""
        from app.langgraph_pipeline import preprocess_node

        state = _state_with_data()
        result = preprocess_node(state)
        X = result["X_full"]
        non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        assert non_numeric == [], f"Non-numeric columns remain: {non_numeric}"

    def test_preprocess_node_no_nulls(self):
        """X_full must not contain any null values after preprocessing."""
        from app.langgraph_pipeline import preprocess_node

        state = _state_with_data()
        result = preprocess_node(state)
        assert result["X_full"].isnull().sum().sum() == 0

    def test_preprocess_node_produces_train_test_split(self):
        """preprocess_node must produce X_train, X_test, y_train, y_test."""
        from app.langgraph_pipeline import preprocess_node

        state = _state_with_data()
        result = preprocess_node(state)
        for key in ("X_train", "X_test", "y_train", "y_test"):
            assert key in result, f"Missing key: {key}"

    def test_preprocess_node_increments_iteration(self):
        """preprocess_node must increment the iteration counter."""
        from app.langgraph_pipeline import preprocess_node

        state = _state_with_data()
        state["iteration"] = 0
        result = preprocess_node(state)
        assert result["iteration"] == 1


# ---------------------------------------------------------------------------
# Tests: train_node
# ---------------------------------------------------------------------------


class TestTrainNode:
    """Tests for train_node."""

    def test_train_node_produces_model_with_predict(self):
        """train_node must return a model with a predict() method."""
        from app.langgraph_pipeline import train_node

        state = _state_after_preprocess()
        result = train_node(state)
        model = result.get("model")
        assert model is not None
        assert hasattr(model, "predict"), "Model must have a predict() method"

    def test_train_node_returns_mae(self):
        """train_node must return a non-negative test_mae."""
        from app.langgraph_pipeline import train_node

        state = _state_after_preprocess()
        result = train_node(state)
        assert "test_mae" in result
        assert result["test_mae"] >= 0.0

    def test_train_node_returns_r2(self):
        """train_node must return a finite test_r2 (can be negative)."""
        from app.langgraph_pipeline import train_node

        state = _state_after_preprocess()
        result = train_node(state)
        assert "test_r2" in result
        assert np.isfinite(result["test_r2"])

    def test_train_node_cv_metrics_present(self):
        """train_node must return cross-validation statistics."""
        from app.langgraph_pipeline import train_node

        state = _state_after_preprocess()
        result = train_node(state)
        for key in ("cv_mae_mean", "cv_mae_std", "cv_r2_mean", "cv_r2_std"):
            assert key in result, f"Missing CV key: {key}"


# ---------------------------------------------------------------------------
# Tests: evaluate_node
# ---------------------------------------------------------------------------


class TestEvaluateNode:
    """Tests for evaluate_node."""

    def test_evaluate_node_returns_confidence_between_0_and_1(self):
        """Confidence score must be in [0, 1]."""
        from app.langgraph_pipeline import evaluate_node

        state = _state_after_train()
        result = evaluate_node(state)
        conf = result.get("confidence", result.get("confidence_score", -1))
        assert 0.0 <= conf <= 1.0, f"confidence={conf} is out of [0, 1]"

    def test_evaluate_node_sets_continue_loop_true_when_below_threshold(self):
        """When confidence < threshold and iterations remain, continue_loop must be True."""
        from app.langgraph_pipeline import evaluate_node

        state = _state_after_train()
        state["confidence_threshold"] = 0.9999  # effectively unreachable
        state["iteration"] = 1
        state["max_iterations"] = 5
        result = evaluate_node(state)
        assert result["continue_loop"] is True

    def test_evaluate_node_sets_continue_loop_false_at_max_iterations(self):
        """When iteration == max_iterations, continue_loop must be False."""
        from app.langgraph_pipeline import evaluate_node

        state = _state_after_train()
        state["confidence_threshold"] = 0.9999
        state["iteration"] = 3
        state["max_iterations"] = 3
        result = evaluate_node(state)
        assert result["continue_loop"] is False

    def test_evaluate_node_appends_to_history(self):
        """evaluate_node must append one entry to state['history']."""
        from app.langgraph_pipeline import evaluate_node

        state = _state_after_train()
        state["history"] = []
        result = evaluate_node(state)
        assert len(result["history"]) == 1

    def test_evaluate_node_returns_metrics_dict(self):
        """evaluate_node must return a 'metrics' dict with expected keys."""
        from app.langgraph_pipeline import evaluate_node

        state = _state_after_train()
        result = evaluate_node(state)
        metrics = result.get("metrics", {})
        assert "test_mae" in metrics
        assert "confidence_score" in metrics


# ---------------------------------------------------------------------------
# Tests: full compiled graph
# ---------------------------------------------------------------------------


class TestCompiledGraph:
    """Tests for the full compiled LangGraph pipeline."""

    def test_full_pipeline_runs_to_completion(self):
        """run_agentic_pipeline must return a non-empty result dict."""
        from app.langgraph_pipeline import run_agentic_pipeline

        result = run_agentic_pipeline(
            dataset_name="uci_student_math",
            confidence_threshold=0.1,  # very easy to meet → single iteration
            max_iterations=1,
        )
        assert isinstance(result, dict)
        assert result, "Result dict must not be empty"

    def test_full_pipeline_returns_expected_keys(self):
        """Result must contain all documented top-level keys."""
        from app.langgraph_pipeline import run_agentic_pipeline

        result = run_agentic_pipeline(
            dataset_name="uci_student_math",
            confidence_threshold=0.5,
            max_iterations=1,
        )
        expected_keys = {
            "dataset",
            "task",
            "model",
            "test_mae",
            "test_r2",
            "confidence_score",
            "confidence_label",
            "iterations",
        }
        missing = expected_keys - set(result.keys())
        assert not missing, f"Missing result keys: {missing}"

    def test_full_pipeline_confidence_within_bounds(self):
        """confidence_score must be in [0, 1]."""
        from app.langgraph_pipeline import run_agentic_pipeline

        result = run_agentic_pipeline(
            dataset_name="uci_student_math",
            confidence_threshold=0.5,
            max_iterations=1,
        )
        score = result["confidence_score"]
        assert 0.0 <= score <= 1.0

    def test_conditional_routing_triggers_retrain_when_threshold_high(self):
        """When threshold=0.99, the loop must exhaust max_iterations."""
        from app.langgraph_pipeline import run_agentic_pipeline

        result = run_agentic_pipeline(
            dataset_name="uci_student_math",
            confidence_threshold=0.99,
            max_iterations=2,
        )
        assert result["iterations"] == 2
        assert result["loop_terminated_reason"] == "max_iterations_reached"

    def test_conditional_routing_stops_early_when_threshold_low(self):
        """When threshold=0.0, the loop must stop after the first iteration."""
        from app.langgraph_pipeline import run_agentic_pipeline

        result = run_agentic_pipeline(
            dataset_name="uci_student_math",
            confidence_threshold=0.0,
            max_iterations=5,
        )
        assert result["iterations"] == 1
        assert result["loop_terminated_reason"] == "confidence_threshold_reached"
