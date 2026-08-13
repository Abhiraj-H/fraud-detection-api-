# 🛡️ Fraud Detection System

End-to-end ML fraud detection with a FastAPI inference service, SHAP explanations, drift monitoring, and a Streamlit dashboard.

## Project Structure

```
fraud-detection-api/
├── train.py                  # Full training pipeline (EDA → XGBoost → SHAP → save)
├── benchmark.py              # p50/p99 latency benchmark
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── api/
│   ├── main.py               # FastAPI: /predict, /drift-check, /health
│   ├── schema.py             # Pydantic schemas (30-feature + SHAP response)
│   └── drift.py              # PSI + KS-test utilities
├── dashboard/
│   └── app.py                # Streamlit: PR curve, cost simulator, drift monitor
├── models/                   # Saved artefacts (after training)
│   ├── model.pkl             # XGBoost
│   ├── lr_model.pkl          # Logistic Regression baseline
│   ├── scaler.pkl            # StandardScaler (Time, Amount)
│   ├── shap_explainer.pkl    # TreeExplainer
│   ├── feature_cols.pkl      # Feature order list
│   ├── train_distributions.pkl  # Reference distributions for drift
│   └── metrics.json          # PR-AUC, ROC-AUC, threshold, PR curve points
├── data/
│   └── creditcard.csv        # ← Download from Kaggle (see below)
├── notebooks/
│   └── 01_eda_and_model.ipynb
└── tests/
    └── test_api.py
```

## Quick Start

### 1. Install dependencies

```bash
cd fraud-detection-api
pip install -r requirements.txt
```

### 2. Get the dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in `data/creditcard.csv`.

### 3. Train the model

```bash
python train.py
```

Output includes:
- EDA summary (class imbalance, feature stats)
- Time-based 70/30 split (no data leakage)
- Logistic Regression baseline metrics
- XGBoost metrics: **PR-AUC**, Precision@Recall=80%, Recall@FPR=1%
- All artefacts saved to `models/`

### 4. Run the API

```bash
uvicorn api.main:app --reload --port 8000
```

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness + model status |
| `POST` | `/predict` | Fraud probability + top-3 SHAP reasons |
| `POST` | `/drift-check` | Per-feature PSI + KS drift report |
| `GET`  | `/docs` | Swagger UI |

#### Example predict request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
    "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
    "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
    "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
    "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
    "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
    "Time": 406.0, "Amount": 149.62
  }'
```

#### Example drift-check request

```bash
curl -X POST http://localhost:8000/drift-check \
  -H "Content-Type: application/json" \
  -d '{"features": {"V1": [-1.2, 0.5, 2.1, -0.3], "Amount": [50, 120, 8, 300]}}'
```

### 5. Run the Dashboard

```bash
streamlit run dashboard/app.py
```

Opens at http://localhost:8501 with three pages:
- **Model Performance** — PR curve, ROC curve, key metrics
- **Cost Simulator** — Confusion matrix + threshold cost sweep
- **Drift Monitor** — Per-feature PSI/KS with distribution overlay

### 6. Latency Benchmark

```bash
# Start the API first, then:
python benchmark.py --n 500
```

Reports p50 / p90 / p99 latency.

### 7. Docker

```bash
docker-compose up --build
```

**Live API:** https://fraud-detection-api-9u8q.onrender.com  
**Docs:** https://fraud-detection-api-9u8q.onrender.com/docs

### 8. Run Tests

```bash
pytest tests/ -v
```

## Metrics

| Metric | Description |
|--------|-------------|
| PR-AUC | Primary metric — area under precision-recall curve |
| Precision@Recall=80% | Precision when recall ≥ 0.80 |
| Recall@FPR=1% | True positive rate at 1% false positive rate |
| ROC-AUC | Secondary metric |

> Plain accuracy is **not** used — meaningless at 0.17% fraud rate.

## Model Design

- **Baseline**: Logistic Regression (`class_weight="balanced"`)
- **Main**: XGBoost with `scale_pos_weight = neg/pos ≈ 577`
- **Split**: Time-based 70/30 (no future data in training)
- **Features**: V1–V28 (PCA), Time + Amount (StandardScaler)
- **Explainability**: SHAP TreeExplainer — top-3 feature reasons per prediction

## Drift Detection

- **PSI** — Population Stability Index per feature
- **KS test** — 2-sample Kolmogorov-Smirnov per feature
- Reference distribution: training data stored in `models/train_distributions.pkl`
- Alert threshold: PSI ≥ 0.25 **or** KS p-value < 0.05
