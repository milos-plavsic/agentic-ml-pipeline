import pytest


def test_finetune_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINETUNE_N_ITER", "3")
    from fastapi.testclient import TestClient

    from app.api import app

    client = TestClient(app)
    r = client.post("/v1/finetune/rf_search")
    assert r.status_code == 200
    data = r.json()
    assert "best_params" in data
    assert data["test_mae"] >= 0
