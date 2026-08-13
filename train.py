"""
Fraud Detection — Full Training Pipeline
=========================================
Day 1: EDA — class imbalance, feature distributions
Day 2-3: Logistic Regression baseline + XGBoost with scale_pos_weight
Metrics: PR-AUC, precision@fixed recall, recall@fixed FPR
SHAP explainability (top-3 per prediction)
Saves: models/model.pkl, models/scaler.pkl, models/feature_cols.pkl, models/metrics.json
       models/train_distributions.pkl  (reference for drift)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "creditcard.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load & EDA
# ─────────────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {CSV_PATH}.\n"
            "Download it from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "and place creditcard.csv in the data/ folder."
        )
    print("Loading dataset ...")
    df = pd.read_csv(CSV_PATH)
    print(f"  Rows: {len(df):,}   Columns: {df.shape[1]}")
    return df


def eda(df: pd.DataFrame) -> None:
    n_fraud = df["Class"].sum()
    n_total = len(df)
    fraud_pct = 100 * n_fraud / n_total
    print(f"\n--- EDA ---")
    print(f"  Total transactions : {n_total:,}")
    print(f"  Fraud cases        : {n_fraud:,}  ({fraud_pct:.4f}%)")
    print(f"  Class imbalance    : 1 fraud per {int(n_total/n_fraud):,} legit txns")
    print(f"  Time range         : {df['Time'].min():.0f}s - {df['Time'].max():.0f}s")
    print(f"  Amount stats       : mean={df['Amount'].mean():.2f}  max={df['Amount'].max():.2f}")
    pca_cols = [c for c in df.columns if c.startswith("V")]
    print(f"  PCA features       : {len(pca_cols)} ({pca_cols[0]}...{pca_cols[-1]})")
    print(f"  Missing values     : {df.isnull().sum().sum()}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Time-based train / test split
# ─────────────────────────────────────────────────────────────────────────────
def split_data(df: pd.DataFrame):
    df = df.sort_values("Time").reset_index(drop=True)
    split_idx = int(len(df) * 0.70)
    train = df.iloc[:split_idx].reset_index(drop=True)
    test  = df.iloc[split_idx:].reset_index(drop=True)
    print(f"Train: {len(train):,} rows  (fraud: {train['Class'].sum():,})")
    print(f"Test : {len(test):,} rows  (fraud: {test['Class'].sum():,})")
    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Feature engineering + scaling
# ─────────────────────────────────────────────────────────────────────────────
PCA_COLS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLS = PCA_COLS + ["Time", "Amount"]


def prepare_features(train: pd.DataFrame, test: pd.DataFrame):
    scaler = StandardScaler()
    X_train = train[FEATURE_COLS].copy()
    X_test  = test[FEATURE_COLS].copy()
    X_train[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
    X_test[["Time", "Amount"]]  = scaler.transform(X_test[["Time", "Amount"]])
    y_train = train["Class"].values
    y_test  = test["Class"].values
    return X_train, y_train, X_test, y_test, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Metric helpers
# ─────────────────────────────────────────────────────────────────────────────
def precision_at_recall(y_true, y_scores, target_recall: float = 0.80) -> float:
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    idx = np.where(rec >= target_recall)[0]
    if len(idx) == 0:
        return 0.0
    return float(prec[idx[0]])


def recall_at_fpr(y_true, y_scores, target_fpr: float = 0.01) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(0, min(idx, len(tpr) - 1))
    return float(tpr[idx])


def threshold_at_fpr(y_true, y_scores, target_fpr: float = 0.01) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(0, min(idx, len(thresholds) - 1))
    return float(thresholds[idx])


def evaluate(name: str, y_true, y_scores) -> dict:
    pr_auc  = average_precision_score(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    p80     = precision_at_recall(y_true, y_scores, 0.80)
    rec1    = recall_at_fpr(y_true, y_scores, 0.01)
    thr1    = threshold_at_fpr(y_true, y_scores, 0.01)
    print(f"\n-- {name} --")
    print(f"  PR-AUC                    : {pr_auc:.4f}")
    print(f"  ROC-AUC                   : {roc_auc:.4f}")
    print(f"  Precision @ Recall=0.80   : {p80:.4f}")
    print(f"  Recall @ FPR=1%           : {rec1:.4f}")
    print(f"  Threshold @ FPR=1%        : {thr1:.4f}")
    return {
        "pr_auc": pr_auc, "roc_auc": roc_auc,
        "precision_at_recall_80": p80,
        "recall_at_fpr_1pct": rec1,
        "threshold_at_fpr_1pct": thr1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Baseline — Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────
def train_logistic(X_train, y_train, X_test, y_test) -> dict:
    print("\n--- Baseline: Logistic Regression ---")
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    scores = lr.predict_proba(X_test)[:, 1]
    metrics = evaluate("Logistic Regression", y_test, scores)
    return {"model": lr, "metrics": metrics, "scores": scores}


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main model — XGBoost
# ─────────────────────────────────────────────────────────────────────────────
def train_xgboost(X_train, y_train, X_test, y_test) -> dict:
    print("\n--- XGBoost ---")
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos
    print(f"  scale_pos_weight = {spw:.1f}")

    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    scores = xgb.predict_proba(X_test)[:, 1]
    metrics = evaluate("XGBoost", y_test, scores)
    return {"model": xgb, "metrics": metrics, "scores": scores}


# ─────────────────────────────────────────────────────────────────────────────
# 7.  SHAP explainer
# ─────────────────────────────────────────────────────────────────────────────
def build_shap_explainer(xgb_model, X_train: pd.DataFrame):
    print("\n--- Building SHAP explainer ---")
    explainer = shap.TreeExplainer(xgb_model)
    sample = X_train.iloc[:5]
    sv = explainer.shap_values(sample)
    print(f"  SHAP values shape: {np.array(sv).shape}  OK")
    return explainer


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Save artefacts
# ─────────────────────────────────────────────────────────────────────────────
def save_artifacts(
    xgb_model,
    lr_model,
    scaler,
    explainer,
    X_train: pd.DataFrame,
    xgb_metrics: dict,
    lr_metrics: dict,
    y_test,
    xgb_scores,
) -> None:
    print("\n--- Saving artefacts ---")

    joblib.dump(xgb_model, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(lr_model,  os.path.join(MODEL_DIR, "lr_model.pkl"))
    joblib.dump(scaler,    os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(explainer, os.path.join(MODEL_DIR, "shap_explainer.pkl"))
    joblib.dump(FEATURE_COLS, os.path.join(MODEL_DIR, "feature_cols.pkl"))

    # Save training feature distributions for drift reference
    train_distributions = {}
    for col in FEATURE_COLS:
        vals = X_train[col].tolist()
        train_distributions[col] = {
            "mean":   float(X_train[col].mean()),
            "std":    float(X_train[col].std()),
            "values": vals,
        }
    joblib.dump(train_distributions, os.path.join(MODEL_DIR, "train_distributions.pkl"))

    # Precision/recall curve points for dashboard
    prec, rec, thr = precision_recall_curve(y_test, xgb_scores)
    step = max(1, len(prec) // 500)

    metrics_out = {
        "xgb_pr_auc":            xgb_metrics["pr_auc"],
        "xgb_roc_auc":           xgb_metrics["roc_auc"],
        "xgb_precision_at_r80":  xgb_metrics["precision_at_recall_80"],
        "xgb_recall_at_fpr1":    xgb_metrics["recall_at_fpr_1pct"],
        "lr_pr_auc":             lr_metrics["pr_auc"],
        "lr_roc_auc":            lr_metrics["roc_auc"],
        "threshold_at_1pct_fpr": xgb_metrics["threshold_at_fpr_1pct"],
        "test_size":             int(len(y_test)),
        "test_fraud_count":      int(y_test.sum()),
        "pr_curve": {
            "precision": prec[::step].tolist(),
            "recall":    rec[::step].tolist(),
        },
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    for name, path in [
        ("XGBoost model",    "model.pkl"),
        ("LR baseline",      "lr_model.pkl"),
        ("Scaler",           "scaler.pkl"),
        ("SHAP explainer",   "shap_explainer.pkl"),
        ("Feature cols",     "feature_cols.pkl"),
        ("Train dists",      "train_distributions.pkl"),
        ("Metrics JSON",     "metrics.json"),
    ]:
        full = os.path.join(MODEL_DIR, path)
        print(f"  OK  {name:25s} -> {full}")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Fraud Detection — Training Pipeline")
    print("=" * 60)

    df = load_data()
    eda(df)
    train, test = split_data(df)
    X_train, y_train, X_test, y_test, scaler = prepare_features(train, test)

    lr_result  = train_logistic(X_train, y_train, X_test, y_test)
    xgb_result = train_xgboost(X_train, y_train, X_test, y_test)

    explainer = build_shap_explainer(xgb_result["model"], X_train)

    save_artifacts(
        xgb_model   = xgb_result["model"],
        lr_model    = lr_result["model"],
        scaler      = scaler,
        explainer   = explainer,
        X_train     = X_train,
        xgb_metrics = xgb_result["metrics"],
        lr_metrics  = lr_result["metrics"],
        y_test      = y_test,
        xgb_scores  = xgb_result["scores"],
    )

    print("\n" + "=" * 60)
    print("  Training complete!")
    print("  Next: uvicorn api.main:app --reload --port 8000")
    print("=" * 60)


if __name__ == "__main__":
    main()
