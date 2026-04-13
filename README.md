# Customer Segmentation MLOps

End-to-end customer segmentation project using unsupervised machine learning with full MLOps practices — from experiment tracking and data versioning to cloud deployment, CI/CD automation, and monitoring.

---

## Dataset

| Property | Details |
|----------|---------|
| Source | Marketing Campaign dataset |
| Size | 2,240 rows × 29 columns |
| Format | Tab-separated CSV |
| Features | Demographics, spending, purchase channels, campaign responses |

---

## Results

**Best Model:** K-Means with 3 clusters

| Cluster | Label | Profile |
|---------|-------|---------|
| 0 | Budget Conscious Families | Low income, many children, price-sensitive |
| 1 | High Value Loyalists | Highest income, max spending, campaign-responsive |
| 2 | Middle Class Actives | Mid-range income, moderate spending across all channels |

---

## Full MLOps Workflow

```
Raw Data
   ↓
Git + DVC          → Version control for code and data
   ↓
MLflow             → Experiment tracking, hyperparameter logging
   ↓
Training Pipeline  → Preprocessing → Feature Engineering → KMeans
   ↓
Model Artifacts    → scaler.pkl, pca.pkl, kmeans_model.pkl
   ↓
FastAPI            → /predict endpoint with Pydantic validation
   ↓
Docker             → Containerized API (Dockerfile.fastapi)
   ↓
Render (Cloud)     → Live public API endpoint
   ↓
GitHub Actions     → CI/CD: test → deploy → retrain → monitor
   ↓
Monitoring         → Data drift + prediction distribution tracking
```

---

## MLOps Progress

| Day | Topic | Deliverable |
|-----|-------|-------------|
| 1 | Project Setup | GitHub repo with folder structure and README |
| 2 | Git & GitHub | Code and model pushed; branches created and merged |
| 3 | DVC Versioning | Dataset and model tracked with `.dvc` files |
| 4 | MLflow Tracking | Experiments logged and compared in MLflow dashboard |
| 5 | Docker (Training) | Docker image loading and running existing model |
| 6 | Flask API | `/predict` endpoint serving cluster predictions locally |
| 7 | Input Validation | 3-level validation with informative error messages |
| 8 | Dockerize Flask API | Flask API containerized; predictions match local results |
| 9 | CI with GitHub Actions | Auto build and test on every push to `main` |
| 10 | End-to-End Integration | Full pipeline verified across local and Docker environments |
| 11 | FastAPI Basics | `/predict` with Pydantic validation; Swagger UI at `/docs` |
| 12 | Dockerize FastAPI | FastAPI containerized; predictions validated |
| 13 | Cloud Deployment | FastAPI deployed live on Render |
| 14 | CI/CD Pipeline | Automated test + deploy + retrain workflows via GitHub Actions |
| 15 | Monitoring | Data drift + prediction distribution; scheduled weekly via Actions |

---

## Tool Summary

| Tool | Purpose | How Used |
|------|---------|----------|
| Git | Code versioning | Feature branches, merge to main |
| DVC | Data/model versioning | `.dvc` files tracking dataset and models |
| MLflow | Experiment tracking | Logged KMeans & Agglomerative runs, silhouette scores |
| Flask | Local API (v1) | `/predict` endpoint with 3-level input validation |
| FastAPI | Production API (v2) | Pydantic validation, Swagger UI, async support |
| Docker | Containerization | `Dockerfile.flask`, `Dockerfile.fastapi` |
| Render | Cloud deployment | Live API at customer-segmentation-fastapi.onrender.com |
| GitHub Actions | CI/CD + Monitoring | `ci.yml`, `deploy.yml`, `retrain.yml`, `monitor.yml` |
| EvidentlyAI | Drift reporting | HTML drift report comparing reference vs current data |
| Scipy (KS Test) | Custom drift detection | p-value based drift detection per feature |

---

## GitHub Actions Workflows

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| `ci.yml` | Every push/PR | Builds FastAPI Docker image, runs API tests locally |
| `deploy.yml` | App code changes | Runs sanity tests, triggers Render deployment |
| `retrain.yml` | Data changes | Retrains KMeans, pushes updated `.pkl` files, redeploys |
| `monitor.yml` | Every Monday 9AM UTC | Runs drift detection, uploads report as artifact |

---

## Monitoring

Two monitoring approaches implemented:

**EvidentlyAI** (`monitoring/monitor.py`)
- Generates HTML drift report
- Uses Wasserstein distance to detect feature drift
- Output: `reports/monitoring/drift_report.html`

**Custom Python Script** (`monitoring/monitor_custom.py`)
- KS Test (Kolmogorov-Smirnov) for drift detection per feature
- Prediction distribution tracking across all 3 clusters
- Alerts when p-value < 0.05

Sample output:
```
DRIFT REPORT
⚠️  DRIFTED | Income     | p-value: 0.0000
⚠️  DRIFTED | MntWines   | p-value: 0.0000
⚠️  DRIFTED | Age        | p-value: 0.0000
✅ OK       | Recency    | p-value: 1.0000
...
Total drifted: 3/25 columns

PREDICTION DISTRIBUTION
Reference → Cluster 0: 42.1%, Cluster 1: 28.6%, Cluster 2: 29.3%
Current   → Cluster 0: 42.3%, Cluster 1: 29.5%, Cluster 2: 28.2%
```

---

## Project Structure

```
customer-segmentation-mlops/
├── .github/workflows/
│   ├── ci.yml              # CI: build & test FastAPI locally
│   ├── deploy.yml          # CD: deploy to Render
│   ├── retrain.yml         # CT: retrain on data changes
│   └── monitor.yml         # Monitoring: weekly drift check
├── data/
│   ├── marketing_campaign.csv
│   └── marketing_data_cleaned.csv
├── models/
│   ├── scaler.pkl
│   ├── pca.pkl
│   └── kmeans_model.pkl
├── monitoring/
│   ├── monitor.py          # EvidentlyAI drift report
│   └── monitor_custom.py   # Custom KS test + prediction drift
├── reports/
│   └── monitoring/
│       └── drift_report.html
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── clustering.py
│   ├── evaluation.py
│   ├── pipeline.py
│   └── log_experiment.py
├── fastapi_app.py          # Production FastAPI
├── flask_app.py            # Legacy Flask API
├── train.py                # Retraining entry point
├── main.py                 # Pipeline entry point
├── test_api.py             # API test script
├── Dockerfile.fastapi      # Docker for FastAPI
├── Dockerfile.flask        # Docker for Flask
├── render.yaml             # Render deployment config
└── requirements.txt
```

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train and save models
python train.py

# Start MLflow dashboard
mlflow ui --port 5001

# Run FastAPI locally
python fastapi_app.py

# Run Dockerized FastAPI
docker build -f Dockerfile.fastapi -t customer-segmentation-fastapi .
docker run -v ${PWD}/models:/app/models -p 8000:8000 customer-segmentation-fastapi

# Run monitoring
python monitoring/monitor_custom.py

# Run EvidentlyAI report
py -3.11 monitoring/monitor.py
```

---

## Live API

**Base URL:** https://customer-segmentation-fastapi.onrender.com

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/predict` | POST | Predict customer segment |
| `/docs` | GET | Swagger UI |

**Sample Request:**
```json
{
  "income": 75000,
  "recency": 20,
  "age": 45,
  "total_children": 1,
  "tenure_days": 400,
  "mnt_wines": 500,
  "marital_status": "Partnered",
  "education_group": "Postgraduate"
}
```

**Sample Response:**
```json
{
  "cluster_id": 1,
  "segment_name": "High Value Loyalists",
  "derived": {
    "total_spending": 850.0,
    "total_purchases": 15,
    "spending_per_purchase": 53.13
  }
}
```

---

## Deployment Status

[![CI](https://github.com/muhammedriswanp/customer-segmentation-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/muhammedriswanp/customer-segmentation-mlops/actions/workflows/ci.yml)
[![Deploy to Render](https://github.com/muhammedriswanp/customer-segmentation-mlops/actions/workflows/deploy.yml/badge.svg)](https://github.com/muhammedriswanp/customer-segmentation-mlops/actions/workflows/deploy.yml)
[![Model Monitoring](https://github.com/muhammedriswanp/customer-segmentation-mlops/actions/workflows/monitor.yml/badge.svg)](https://github.com/muhammedriswanp/customer-segmentation-mlops/actions/workflows/monitor.yml)