from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200


def test_pipeline_run() -> None:
    r = client.post("/v1/pipeline/run", json={"dataset_name": "demo.csv"})
    assert r.status_code == 200
    data = r.json()
    assert data["dataset"] == "demo.csv"
    assert "best_model" in data
