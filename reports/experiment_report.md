# Customer Segmentation MLOps

End-to-end customer segmentation project using unsupervised machine learning with MLOps practices including experiment tracking, data versioning, and deployment.

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
| 0 | 🟣 Budget Conscious Families | Low income, many children, price-sensitive |
| 1 | 🟡 High Value Loyalists | Highest income, max spending, campaign-responsive |
| 2 | 🔵 Middle Class Actives | Mid-range income, moderate spending across all channels |

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
| 8 | Dockerize API | Flask API containerized; predictions match local results |
| 9 | CI with GitHub Actions | Auto build and test on every push to `main` |
| 10 | End-to-End Integration | Full pipeline verified across local and Docker environments |

---

## Project Structure
```
customer-segmentation-mlops/
├── .github/workflows/ci.yml        # GitHub Actions CI pipeline
├── data/
│   ├── raw/                        # Original dataset
│   └── processed/                  # Cleaned & clustered output
├── models/                         # Saved model artifacts
├── notebooks/                      # EDA notebook
├── reports/                        # Experiment report
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── clustering.py
│   ├── evaluation.py
│   ├── pipeline.py
│   └── log_experiment.py
├── flask_app.py                    # Flask prediction API
├── test_api.py                     # API test script
├── Dockerfile.flask                # Dockerfile for Flask API
├── app.py                          # Streamlit app
├── main.py                         # Pipeline entry point
└── requirements.txt
```

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train and save models
python main.py

# Start MLflow dashboard
mlflow ui

# Run Flask API
python flask_app.py

# Run Dockerized API
docker build -f Dockerfile.flask -t customer-segmentation-api .
docker run -v ${PWD}/models:/app/models -p 5000:5000 customer-segmentation-api

# Run Streamlit app
python -m streamlit run app.py
```