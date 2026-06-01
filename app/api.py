"""FastAPI application for the Agentic ML Pipeline.

Endpoints
---------
POST /v1/pipeline/run   — runs the full LangGraph pipeline, returns metrics + report
GET  /v1/pipeline/status — current (or most recent) run status from in-memory store
GET  /v1/datasets        — list available datasets
GET  /health             — liveness probe
GET  /metrics            — Prometheus metrics
"""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ml_core import (
    APIKeyMiddleware,
    RateLimiter,
    RateLimitExceeded,
    configure_logging,
    install_middleware,
)
from ml_core.observability import metrics_router, observe_request
from pydantic import BaseModel, Field

from app.main import run_pipeline
from app.studio import datasets as studio_datasets
from app.studio.mlflow_tracker import (
    list_recent_runs,
    log_studio_run,
    tracking_status,
)
from app.studio.registry import list_plugins, run_plugin
from finetune.tuner import run_rf_hyperparam_finetune

logger = configure_logging("api")

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AutoML / Benchmark Studio",
    version="2.0.0",
    description=(
        "LangGraph AutoML pipeline plus merged tabular plugins "
        "(ensemble benchmark, neural regression) with dataset upload."
    ),
)

_ui_dir = Path(__file__).resolve().parent / "static"
if _ui_dir.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_ui_dir), html=True), name="studio-ui")

# Middleware: request IDs, security headers, CORS
install_middleware(app, cors_allow_origins=("*",))

# API-key auth (no-op when API_KEY env var is unset — dev mode)
app.add_middleware(APIKeyMiddleware)

# Prometheus metrics endpoint
app.include_router(metrics_router)

# Per-IP rate limiter: 20 req/s, burst 40
_limiter = RateLimiter(rate=20.0, burst=40.0)

# ---------------------------------------------------------------------------
# In-memory run-status store
# ---------------------------------------------------------------------------

_status_lock = threading.Lock()
_pipeline_status: dict[str, Any] = {
    "run_id": None,
    "status": "idle",  # idle | running | completed | failed
    "started_at": None,
    "completed_at": None,
    "error": None,
    "result": None,
}


def _get_client_key(request: Request) -> str:
    """Return a stable per-client key for rate limiting."""
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


async def rate_limit_dep(request: Request) -> None:
    """FastAPI dependency that enforces per-client rate limiting."""
    key = _get_client_key(request)
    try:
        _limiter.acquire(key)
    except RateLimitExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class TrainRequest(BaseModel):
    """Request body for POST /v1/pipeline/run."""

    dataset_name: str = Field(
        "uci_student_math",
        description="Only `uci_student_math` is supported (UCI student-mat.csv).",
    )
    confidence_threshold: float = Field(
        0.68,
        ge=0.0,
        le=1.0,
        description="Confidence target used by the LangGraph retry loop.",
    )
    max_iterations: int = Field(
        3,
        ge=1,
        le=8,
        description="Max refinement iterations before returning best available result.",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/v1/datasets", tags=["pipeline"], dependencies=[Depends(rate_limit_dep)])
async def list_datasets() -> dict[str, Any]:
    """List built-in and uploaded datasets."""
    builtin = [
        {
            "dataset_id": "uci_student_math",
            "name": "uci_student_math",
            "description": (
                "UCI Student Performance - Mathematics (secondary school, Portugal). "
                "395 rows x 33 features. Target: G3 final grade (0-20)."
            ),
            "source": "https://archive.ics.uci.edu/dataset/320/student+performance",
            "task": "regression",
            "target_column": "G3",
        }
    ]
    return {"datasets": builtin + studio_datasets.list_datasets()}


@app.post("/v1/datasets/upload", tags=["studio"], dependencies=[Depends(rate_limit_dep)])
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload a CSV for studio plugins (max ~5MB in memory)."""
    content = await file.read()
    if len(content) > 5_000_000:
        raise HTTPException(status_code=413, detail="file too large (max 5MB)")
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="only .csv uploads supported")
    return studio_datasets.register_upload(file.filename, content)


@app.get("/v1/studio/plugins", tags=["studio"], dependencies=[Depends(rate_limit_dep)])
async def studio_plugins() -> dict[str, Any]:
    return {"plugins": list_plugins()}


class StudioRunRequest(BaseModel):
    plugin_id: str = Field(..., min_length=1)
    dataset_id: str | None = None
    dataset_name: str = "uci_student_math"
    cv_splits: int = Field(3, ge=2, le=10)
    confidence_threshold: float = Field(0.68, ge=0.0, le=1.0)
    max_iterations: int = Field(3, ge=1, le=8)
    epochs: int = Field(40, ge=5, le=200)


@app.post("/v1/studio/run", tags=["studio"], dependencies=[Depends(rate_limit_dep)])
async def studio_run(body: StudioRunRequest) -> dict[str, Any]:
    """Run a registered studio plugin on an uploaded or built-in dataset."""
    try:
        if body.plugin_id == "automl_pipeline":
            result = run_plugin(
                body.plugin_id,
                dataset_name=body.dataset_name,
                confidence_threshold=body.confidence_threshold,
                max_iterations=body.max_iterations,
            )
            run_id = log_studio_run(body.plugin_id, result)
            if run_id:
                result["mlflow_run_id"] = run_id
            return result
        if not body.dataset_id:
            raise HTTPException(status_code=422, detail="dataset_id required for this plugin")
        if body.plugin_id == "nn_regression":
            result = run_plugin(
                body.plugin_id,
                dataset_id=body.dataset_id,
                epochs=body.epochs,
            )
        else:
            result = run_plugin(
                body.plugin_id,
                dataset_id=body.dataset_id,
                cv_splits=body.cv_splits,
            )
        run_id = log_studio_run(body.plugin_id, result, dataset_id=body.dataset_id)
        if run_id:
            result["mlflow_run_id"] = run_id
        return result
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/ui", tags=["studio"], include_in_schema=False)
async def studio_ui_redirect() -> FileResponse:
    index = _ui_dir / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="UI not bundled")
    return FileResponse(index)


@app.post("/v1/pipeline/run", tags=["pipeline"], dependencies=[Depends(rate_limit_dep)])
async def run_train(request: Request, body: TrainRequest) -> dict:
    """Run the full LangGraph pipeline and return metrics + report.

    The pipeline uses a confidence-based retry loop: if the model does not
    reach `confidence_threshold` it will re-train (up to `max_iterations`
    times) with richer features before returning.
    """
    run_id = uuid.uuid4().hex

    with _status_lock:
        _pipeline_status.update(
            {
                "run_id": run_id,
                "status": "running",
                "started_at": time.time(),
                "completed_at": None,
                "error": None,
                "result": None,
            }
        )

    try:
        result = run_pipeline(
            body.dataset_name,
            confidence_threshold=body.confidence_threshold,
            max_iterations=body.max_iterations,
        )
        with _status_lock:
            _pipeline_status.update(
                {
                    "status": "completed",
                    "completed_at": time.time(),
                    "result": result,
                }
            )
        logger.info(
            "Pipeline run %s completed: confidence=%s",
            run_id,
            result.get("confidence_score"),
        )
        return {"run_id": run_id, **result}

    except Exception as exc:
        logger.error(f"Pipeline run {run_id} failed: {exc}")
        with _status_lock:
            _pipeline_status.update(
                {
                    "status": "failed",
                    "completed_at": time.time(),
                    "error": str(exc),
                }
            )
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc


@app.get("/v1/pipeline/status", tags=["pipeline"], dependencies=[Depends(rate_limit_dep)])
async def pipeline_status() -> dict[str, Any]:
    """Return the current or most recent pipeline run status."""
    with _status_lock:
        snapshot = dict(_pipeline_status)

    # Don't embed the full result in the status response — callers get that
    # from the /run response. Include a summary only.
    result = snapshot.pop("result", None)
    if result:
        snapshot["summary"] = {
            "test_mae": result.get("test_mae"),
            "test_r2": result.get("test_r2"),
            "confidence_score": result.get("confidence_score"),
            "iterations": result.get("iterations"),
        }
    return snapshot


@app.get("/v1/studio/mlflow/status", tags=["studio"])
async def studio_mlflow_status() -> dict:
    return tracking_status()


@app.get("/v1/studio/experiments", tags=["studio"])
async def studio_experiments(limit: int = 20) -> dict:
    runs = list_recent_runs(max_results=min(limit, 100))
    return {"runs": runs, "count": len(runs)}


@app.post("/v1/finetune/rf_search", tags=["finetune"], dependencies=[Depends(rate_limit_dep)])
async def finetune_rf_search() -> dict:
    """Run random-forest hyper-parameter search and return best params."""
    return run_rf_hyperparam_finetune()


# ---------------------------------------------------------------------------
# Prometheus request-observation middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def _observe(request: Request, call_next):
    return await observe_request(request, call_next)
