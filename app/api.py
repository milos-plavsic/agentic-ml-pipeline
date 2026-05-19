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
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
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
from finetune.tuner import run_rf_hyperparam_finetune

logger = configure_logging("api")

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Agentic ML Pipeline",
    version="1.0.0",
    description=(
        "LangGraph-orchestrated RandomForest/GradientBoosting pipeline "
        "for the UCI Student Performance dataset."
    ),
)

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
    """List datasets that the pipeline supports."""
    return {
        "datasets": [
            {
                "name": "uci_student_math",
                "description": (
                    "UCI Student Performance - Mathematics (secondary school, Portugal). "
                    "395 rows x 33 features. Target: G3 final grade (0-20)."
                ),
                "source": ("https://archive.ics.uci.edu/dataset/320/student+performance"),
                "task": "regression",
                "target_column": "G3",
            }
        ]
    }


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
        logger.info(f"Pipeline run {run_id} completed: confidence={result.get('confidence_score')}")
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
