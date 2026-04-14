from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.main import run_pipeline
from finetune.tuner import run_rf_hyperparam_finetune

app = FastAPI(title="Agentic ML Pipeline", version="0.1.0")


class TrainRequest(BaseModel):
    dataset_name: str = Field(
        "uci_student_math",
        description="Only `uci_student_math` is supported (UCI student-mat.csv).",
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/pipeline/run")
def run_train(body: TrainRequest) -> dict:
    return run_pipeline(body.dataset_name)


@app.post("/v1/finetune/rf_search")
def finetune_rf_search() -> dict:
    return run_rf_hyperparam_finetune()
