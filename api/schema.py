"""Pydantic schemas for the Fraud Detection API."""

from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

class TransactionFeatures(BaseModel):
    """
    Full feature set for a credit-card transaction.
    V1-V28  — PCA-transformed features (anonymized)
    Time    — seconds elapsed since the first transaction in the dataset
    Amount  — transaction amount
    """
    V1:     float
    V2:     float
    V3:     float
    V4:     float
    V5:     float
    V6:     float
    V7:     float
    V8:     float
    V9:     float
    V10:    float
    V11:    float
    V12:    float
    V13:    float
    V14:    float
    V15:    float
    V16:    float
    V17:    float
    V18:    float
    V19:    float
    V20:    float
    V21:    float
    V22:    float
    V23:    float
    V24:    float
    V25:    float
    V26:    float
    V27:    float
    V28:    float
    Time:   float = Field(..., description="Seconds since first transaction")
    Amount: float = Field(..., ge=0.0, description="Transaction amount (USD)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
                "V5": -0.34, "V6": 0.46,  "V7": 0.24, "V8": 0.10,
                "V9": 0.36,  "V10": 0.09, "V11": -0.55, "V12": -0.62,
                "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
                "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
                "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
                "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
                "Time": 406.0, "Amount": 149.62
            }
        }
    }


class ShapReason(BaseModel):
    feature:    str
    value:      float
    shap_value: float
    direction:  str   # "increases_fraud_risk" | "decreases_fraud_risk"


class PredictionResponse(BaseModel):
    is_fraud:      bool
    probability:   float = Field(..., ge=0.0, le=1.0)
    threshold:     float
    model_version: str
    shap_reasons:  List[ShapReason] = Field(
        default_factory=list,
        description="Top-3 SHAP feature contributions"
    )


# ---------------------------------------------------------------------------
# Drift check  (POST /drift-check)
# ---------------------------------------------------------------------------

class FeatureBatch(BaseModel):
    """A batch of incoming transactions as a dict of feature -> list of values."""
    features: dict = Field(
        ...,
        description="Dict mapping feature name to list of observed values. "
                    "Use the same feature names as TransactionFeatures."
    )
    n_samples: Optional[int] = Field(
        None,
        description="Optional hint for how many samples are in the batch."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": {
                    "V1":     [-1.2, 0.5, 2.1, -0.3],
                    "Amount": [50.0, 120.0, 8.5, 300.0],
                }
            }
        }
    }


class FeatureDriftResult(BaseModel):
    feature:       str
    psi:           float
    ks_statistic:  float
    ks_p_value:    float
    drift_detected: bool


class DriftCheckResponse(BaseModel):
    features_checked:  int
    features_drifted:  int
    overall_drift:     bool
    results:           List[FeatureDriftResult]


# ---------------------------------------------------------------------------
# Legacy flat schema (kept for backward-compat unit tests)
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """Flat feature list — legacy, deprecated in favour of TransactionFeatures."""
    features: List[float] = Field(
        ..., min_length=30, max_length=30,
        description="[V1..V28, Time, Amount] in order"
    )


class DriftRequest(BaseModel):
    reference_data: List[float]
    current_data:   List[float]
