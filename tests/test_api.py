import math

from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200


def test_pipeline_run() -> None:
    r = client.post(
        "/v1/pipeline/run",
        json={
            "dataset_name": "uci_student_math",
            "confidence_threshold": 0.75,
            "max_iterations": 3,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["dataset"] == "uci_student_math"
    assert data["task"] == "regression"
    assert data["test_mae"] >= 0
    assert math.isfinite(data["test_r2"])
    assert data["n_rows"] > 50
    assert 0.0 <= data["confidence_score"] <= 1.0
    assert data["confidence_label"] in {"low", "medium", "high"}
    assert 1 <= data["iterations"] <= 3
    assert len(data["iteration_history"]) == data["iterations"]
