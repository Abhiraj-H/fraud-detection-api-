"""
Unit tests for the Fraud Detection API.
Run:  pytest tests/test_api.py -v
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from api.main import app

# Use lifespan context so startup loads model artifacts
client = TestClient(app, raise_server_exceptions=True)

@pytest.fixture(scope="session", autouse=True)
def start_app():
    with TestClient(app) as c:
        # Replace module-level client with the lifespan-aware one
        import tests.test_api as self_mod
        self_mod.client = c
        yield
        self_mod.client = TestClient(app)

# Full 30-feature sample transaction (V1..V28, Time, Amount)
SAMPLE_TX = {
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
    "V5": -0.34, "V6": 0.46,  "V7": 0.24, "V8": 0.10,
    "V9": 0.36,  "V10": 0.09, "V11": -0.55, "V12": -0.62,
    "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
    "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
    "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
    "Time": 406.0, "Amount": 149.62,
}


# ─────────────────────────────────────────────────────────────────────────────
# /health
# ─────────────────────────────────────────────────────────────────────────────

def test_health_returns_200():
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_schema():
    data = client.get("/health").json()
    assert "status" in data
    assert "model_loaded" in data
    assert data["status"] in ("healthy", "degraded")


# ─────────────────────────────────────────────────────────────────────────────
# /predict
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_returns_200_or_503():
    resp = client.post("/predict", json=SAMPLE_TX)
    # 200 if model is loaded, 503 if not (CI without model.pkl)
    assert resp.status_code in (200, 503)


def test_predict_schema_when_model_loaded():
    resp = client.post("/predict", json=SAMPLE_TX)
    if resp.status_code == 503:
        pytest.skip("Model not loaded — skip schema test")
    data = resp.json()
    assert "is_fraud" in data
    assert "probability" in data
    assert "threshold" in data
    assert "model_version" in data
    assert "shap_reasons" in data
    assert isinstance(data["is_fraud"], bool)
    assert 0.0 <= data["probability"] <= 1.0
    assert isinstance(data["shap_reasons"], list)


def test_predict_shap_reasons_structure():
    resp = client.post("/predict", json=SAMPLE_TX)
    if resp.status_code == 503:
        pytest.skip("Model not loaded")
    reasons = resp.json()["shap_reasons"]
    for r in reasons:
        assert "feature" in r
        assert "value" in r
        assert "shap_value" in r
        assert "direction" in r
        assert r["direction"] in ("increases_fraud_risk", "decreases_fraud_risk")


def test_predict_missing_feature_422():
    """Missing a required field → 422 Unprocessable Entity."""
    bad = {k: v for k, v in SAMPLE_TX.items() if k != "Amount"}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_negative_amount_422():
    """Amount must be >= 0."""
    bad = {**SAMPLE_TX, "Amount": -10.0}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# /drift-check  (new endpoint)
# ─────────────────────────────────────────────────────────────────────────────

DRIFT_PAYLOAD = {
    "features": {
        "V1":     list(np.random.normal(0, 1, 200).tolist()),
        "Amount": list(np.random.normal(100, 50, 200).tolist()),
    }
}


def test_drift_check_returns_200():
    resp = client.post("/drift-check", json=DRIFT_PAYLOAD)
    assert resp.status_code == 200


def test_drift_check_schema():
    resp = client.post("/drift-check", json=DRIFT_PAYLOAD)
    if resp.status_code != 200:
        pytest.skip("Drift endpoint unavailable")
    data = resp.json()
    assert "features_checked" in data
    assert "features_drifted" in data
    assert "overall_drift" in data
    assert "results" in data
    for r in data["results"]:
        assert "feature" in r
        assert "psi" in r
        assert "ks_statistic" in r
        assert "ks_p_value" in r
        assert "drift_detected" in r


# ─────────────────────────────────────────────────────────────────────────────
# /drift  (legacy)
# ─────────────────────────────────────────────────────────────────────────────

def test_legacy_drift_returns_200():
    payload = {
        "reference_data": list(np.random.normal(0, 1, 100).tolist()),
        "current_data":   list(np.random.normal(0, 1, 100).tolist()),
    }
    resp = client.post("/drift", json=payload)
    assert resp.status_code == 200


def test_legacy_drift_schema():
    payload = {
        "reference_data": list(np.random.normal(0, 1, 100).tolist()),
        "current_data":   list(np.random.normal(0.5, 1.2, 100).tolist()),
    }
    data = client.post("/drift", json=payload).json()
    assert "psi" in data
    assert "ks_statistic" in data
    assert "ks_p_value" in data
    assert "drift_detected" in data
    assert isinstance(data["drift_detected"], bool)


# ─────────────────────────────────────────────────────────────────────────────
# /predict-legacy
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_legacy():
    features = [SAMPLE_TX[f"V{i}"] for i in range(1, 29)] + [
        SAMPLE_TX["Time"], SAMPLE_TX["Amount"]
    ]
    resp = client.post("/predict-legacy", json={"features": features})
    assert resp.status_code in (200, 503)
