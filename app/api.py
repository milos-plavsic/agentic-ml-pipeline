from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.main import run_pipeline
from finetune.tuner import run_rf_hyperparam_finetune

app = FastAPI(title="Agentic ML Pipeline", version="0.2.0")


class TrainRequest(BaseModel):
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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/pipeline/run")
def run_train(body: TrainRequest) -> dict:
    return run_pipeline(
        body.dataset_name,
        confidence_threshold=body.confidence_threshold,
        max_iterations=body.max_iterations,
    )


@app.post("/v1/finetune/rf_search")
def finetune_rf_search() -> dict:
    return run_rf_hyperparam_finetune()
