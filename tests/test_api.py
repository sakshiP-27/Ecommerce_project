"""Tests for the FastAPI /predict endpoint (Week 3 Step 9)."""


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_expected_fields(client, sample_session):
    response = client.post("/predict", json=sample_session)
    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] in ("Purchase", "No Purchase")
    assert 0.0 <= data["probability"] <= 1.0
    assert "top_features" in data
    assert isinstance(data["top_features"], list)
    assert len(data["top_features"]) > 0


def test_predict_handles_unknown_category(client, sample_session):
    # Step 8 edge case: Month never seen in training
    payload = {**sample_session, "Month": "Smarch"}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] in ("Purchase", "No Purchase")
