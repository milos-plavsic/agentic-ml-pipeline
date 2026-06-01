from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.studio.plugins.ensemble import run_ensemble_benchmark
from app.studio.plugins.nn_regression import run_nn_regression

PluginFn = Callable[..., dict[str, Any]]

_REGISTRY: dict[str, dict[str, object]] = {
    "ensemble_benchmark": {
        "title": "Ensemble benchmark (RF / XGB / CatBoost)",
        "description": "Cross-validated RF, XGBoost, and CatBoost benchmarks",
        "fn": run_ensemble_benchmark,
    },
    "nn_regression": {
        "title": "PyTorch MLP regression",
        "description": "PyTorch MLP regression baseline",
        "fn": run_nn_regression,
    },
    "automl_pipeline": {
        "title": "LangGraph AutoML pipeline",
        "description": "Original agentic-ml-pipeline graph (uci_student_math only)",
        "fn": None,
    },
}


def list_plugins() -> list[dict[str, str]]:
    return [
        {
            "id": pid,
            "title": str(meta["title"]),
            "description": str(meta["description"]),
        }
        for pid, meta in _REGISTRY.items()
    ]


def run_plugin(plugin_id: str, **kwargs: Any) -> dict[str, Any]:
    if plugin_id not in _REGISTRY:
        raise KeyError(f"unknown plugin: {plugin_id}")
    fn = _REGISTRY[plugin_id]["fn"]
    if fn is None:
        from app.main import run_pipeline

        return run_pipeline(
            kwargs.get("dataset_name", "uci_student_math"),
            confidence_threshold=float(kwargs.get("confidence_threshold", 0.68)),
            max_iterations=int(kwargs.get("max_iterations", 3)),
        )
    return fn(**kwargs)
