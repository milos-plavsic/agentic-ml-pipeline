from __future__ import annotations

import os
from typing import Any


def _configure() -> bool:
    try:
        import mlflow
    except ImportError:
        return False
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if uri:
        mlflow.set_tracking_uri(uri)
    registry = os.environ.get("MLFLOW_REGISTRY_URI", "").strip()
    if registry:
        mlflow.set_registry_uri(registry)
    return True


def mlflow_enabled() -> bool:
    return _configure()


def log_studio_run(
    plugin_id: str,
    result: dict[str, Any],
    *,
    dataset_id: str | None = None,
    params: dict[str, Any] | None = None,
) -> str | None:
    """Record studio metrics, params, and a JSON artifact in MLflow."""
    try:
        import mlflow
    except ImportError:
        return None

    if not _configure():
        return None

    experiment = os.environ.get("MLFLOW_EXPERIMENT", "automl-studio")
    mlflow.set_experiment(experiment)
    tags = {"plugin_id": plugin_id}
    if dataset_id:
        tags["dataset_id"] = dataset_id

    with mlflow.start_run(
        run_name=f"{plugin_id}-{dataset_id or 'builtin'}",
        tags=tags,
    ):
        mlflow.log_param("plugin_id", plugin_id)
        if dataset_id:
            mlflow.log_param("dataset_id", dataset_id)
        for key, value in (params or {}).items():
            mlflow.log_param(key, value)
        if "leaderboard" in result:
            for row in result["leaderboard"]:
                model = str(row.get("model", "model"))
                metric = str(row.get("metric", "score"))
                mlflow.log_metric(f"{model}_{metric}", float(row["mean"]))
        if "test_mae" in result:
            mlflow.log_metric("test_mae", float(result["test_mae"]))
        if "val_r2" in result:
            mlflow.log_metric("val_r2", float(result["val_r2"]))
        mlflow.log_dict(result, "studio_result.json")
        run = mlflow.active_run()
        return run.info.run_id if run else None


def list_recent_runs(
    *,
    experiment: str | None = None,
    max_results: int = 20,
) -> list[dict[str, Any]]:
    """Return recent MLflow runs for the studio experiment."""
    try:
        import mlflow
        from mlflow.entities import ViewType
    except ImportError:
        return []

    if not _configure():
        return []

    exp_name = experiment or os.environ.get("MLFLOW_EXPERIMENT", "automl-studio")
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        return []
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=max_results,
        order_by=["start_time DESC"],
    )
    out: list[dict[str, Any]] = []
    for run in runs:
        out.append(
            {
                "run_id": run.info.run_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "params": dict(run.data.params),
                "metrics": {k: v for k, v in run.data.metrics.items()},
                "tags": dict(run.data.tags),
            }
        )
    return out


def tracking_status() -> dict[str, Any]:
    """Surface whether MLflow is installed and which URI is configured."""
    enabled = mlflow_enabled()
    return {
        "enabled": enabled,
        "tracking_uri": os.environ.get("MLFLOW_TRACKING_URI", "local ./mlruns"),
        "experiment": os.environ.get("MLFLOW_EXPERIMENT", "automl-studio"),
    }
