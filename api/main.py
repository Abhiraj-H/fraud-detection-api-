"""
FastAPI — Fraud Detection Inference Service
Endpoints:
  GET  /health           — liveness + model status
  POST /predict          — fraud probability + SHAP top-3 reasons
  POST /drift-check      — per-feature PSI + KS drift report
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from api.schema import (
    TransactionFeatures,
    PredictionResponse,
    ShapReason,
    FeatureBatch,
    DriftCheckResponse,
    FeatureDriftResult,
    # Legacy
    PredictionRequest,
    DriftRequest,
)
from api.drift import calculate_psi, calculate_ks

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud-api")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "models")

FEATURE_ORDER = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]

# Global state
_state: dict = {}

MODEL_VERSION = "1.0.0"
DEFAULT_THRESHOLD = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Startup / shutdown
# ─────────────────────────────────────────────────────────────────────────────

def _load_artifact(filename: str):
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        return joblib.load(path)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model artefacts ...")
    _state["model"]          = _load_artifact("model.pkl")
    _state["scaler"]         = _load_artifact("scaler.pkl")
    _state["explainer"]      = _load_artifact("shap_explainer.pkl")
    _state["feature_cols"]   = _load_artifact("feature_cols.pkl") or FEATURE_ORDER
    _state["train_dists"]    = _load_artifact("train_distributions.pkl") or {}

    # Read decision threshold from saved metrics
    import json
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        _state["threshold"] = m.get("threshold_at_1pct_fpr", DEFAULT_THRESHOLD)
    else:
        _state["threshold"] = DEFAULT_THRESHOLD

    if _state["model"] is not None:
        logger.info(f"Model loaded: {type(_state['model']).__name__}")
    else:
        logger.warning("No model found — /predict will return 503")

    yield
    _state.clear()


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit-card fraud scoring with SHAP explanations and drift monitoring.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Middleware — latency logging
# ─────────────────────────────────────────────────────────────────────────────

@app.middleware("http")
async def add_process_time(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-Ms"] = f"{ms:.2f}"
    logger.debug(f"{request.method} {request.url.path}  {ms:.1f}ms")
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _require_model():
    if _state.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")


def _transaction_to_array(tx: TransactionFeatures) -> np.ndarray:
    """Convert a TransactionFeatures object → scaled numpy row (1 x 30)."""
    raw = np.array([getattr(tx, col) for col in FEATURE_ORDER], dtype=float)
    scaler = _state.get("scaler")
    if scaler is not None:
        # Only Time and Amount were scaled during training
        raw[-2] = scaler.transform([[raw[-2], raw[-1]]])[0][0]
        raw[-1] = scaler.transform([[raw[-2], raw[-1]]])[0][1]
    return raw.reshape(1, -1)


def _scale_row(raw: np.ndarray) -> np.ndarray:
    """Scale a 1x30 raw feature array using the saved scaler (Time, Amount only)."""
    scaler = _state.get("scaler")
    row = raw.copy()
    if scaler is not None:
        import pandas as pd
        ta_df = pd.DataFrame(row[:, -2:], columns=["Time", "Amount"])
        row[:, -2:] = scaler.transform(ta_df)
    return row


def _get_shap_reasons(features_row: np.ndarray, top_n: int = 3) -> list[ShapReason]:
    """Compute SHAP values and return top_n most impactful features."""
    explainer = _state.get("explainer")
    if explainer is None:
        return []
    try:
        sv = explainer.shap_values(features_row)
        # sv may be 2D [1, n_features] for tree explainer
        sv_flat = np.array(sv).flatten()
        feat_names = _state["feature_cols"]
        top_idx = np.argsort(np.abs(sv_flat))[::-1][:top_n]
        reasons = []
        for i in top_idx:
            val = float(features_row[0, i])
            shap_val = float(sv_flat[i])
            reasons.append(ShapReason(
                feature=feat_names[i],
                value=round(val, 4),
                shap_value=round(shap_val, 4),
                direction="increases_fraud_risk" if shap_val > 0 else "decreases_fraud_risk",
            ))
        return reasons
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Meta"])
def root():
    return {
        "service": "Fraud Detection API",
        "version": MODEL_VERSION,
        "model_loaded": _state.get("model") is not None,
        "docs": "/docs",
    }


@app.get("/health", tags=["Meta"])
def health():
    model_ok = _state.get("model") is not None
    return {
        "status": "healthy" if model_ok else "degraded",
        "model_loaded": model_ok,
        "model_type": type(_state["model"]).__name__ if model_ok else "none",
        "threshold": _state.get("threshold", DEFAULT_THRESHOLD),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(tx: TransactionFeatures):
    """
    Score a single transaction.
    Returns fraud probability, binary decision, and top-3 SHAP feature reasons.
    """
    _require_model()
    try:
        raw = np.array([getattr(tx, col) for col in FEATURE_ORDER], dtype=float).reshape(1, -1)
        features_scaled = _scale_row(raw)

        prob      = float(_state["model"].predict_proba(features_scaled)[0][1])
        threshold = _state.get("threshold", DEFAULT_THRESHOLD)
        is_fraud  = prob >= threshold

        reasons = _get_shap_reasons(features_scaled)

        return PredictionResponse(
            is_fraud=is_fraud,
            probability=round(prob, 6),
            threshold=round(threshold, 6),
            model_version=MODEL_VERSION,
            shap_reasons=reasons,
        )
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@app.post("/drift-check", response_model=DriftCheckResponse, tags=["Monitoring"])
def drift_check(batch: FeatureBatch):
    """
    Compare a batch of incoming feature values against the training distribution.
    Returns per-feature PSI + KS statistics.
    """
    train_dists = _state.get("train_dists", {})
    results: list[FeatureDriftResult] = []

    for feature, incoming_vals in batch.features.items():
        if feature not in train_dists:
            continue  # Unknown feature — skip silently
        ref_vals = np.array(train_dists[feature]["values"], dtype=float)
        cur_vals = np.array(incoming_vals, dtype=float)

        psi = calculate_psi(ref_vals, cur_vals)
        ks  = calculate_ks(ref_vals, cur_vals)

        drifted = (psi >= 0.25) or (ks["p_value"] < 0.05)
        results.append(FeatureDriftResult(
            feature=feature,
            psi=round(psi, 6),
            ks_statistic=round(ks["statistic"], 6),
            ks_p_value=round(ks["p_value"], 6),
            drift_detected=drifted,
        ))

    n_drifted = sum(1 for r in results if r.drift_detected)
    return DriftCheckResponse(
        features_checked=len(results),
        features_drifted=n_drifted,
        overall_drift=n_drifted > 0,
        results=results,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Legacy endpoints (backward compat)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/predict-legacy", tags=["Legacy"])
def predict_legacy(request: PredictionRequest):
    """Flat 30-float list → fraud probability (no SHAP). Kept for test suites."""
    _require_model()
    try:
        arr = np.array(request.features, dtype=float).reshape(1, -1)
        arr = _scale_row(arr)
        prob     = float(_state["model"].predict_proba(arr)[0][1])
        threshold = _state.get("threshold", DEFAULT_THRESHOLD)
        return {
            "is_fraud":      prob >= threshold,
            "probability":   round(prob, 6),
            "model_version": MODEL_VERSION,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/drift", tags=["Legacy"])
def drift_legacy(request: DriftRequest):
    """Single-vector PSI + KS comparison (legacy endpoint)."""
    try:
        ref = np.array(request.reference_data, dtype=float)
        cur = np.array(request.current_data, dtype=float)
        psi = calculate_psi(ref, cur)
        ks  = calculate_ks(ref, cur)
        return {
            "psi":            psi,
            "ks_statistic":   ks["statistic"],
            "ks_p_value":     ks["p_value"],
            "drift_detected": psi >= 0.25 or ks["p_value"] < 0.05,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
