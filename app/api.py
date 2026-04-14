from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.main import run_pipeline

app = FastAPI(title="Agentic ML Pipeline", version="0.1.0")


class TrainRequest(BaseModel):
    dataset_name: str = Field(..., min_length=1, description="Dataset identifier or path")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/pipeline/run")
def run_train(body: TrainRequest) -> dict:
    return run_pipeline(body.dataset_name)
