# Customer Segmentation MLOps

End-to-end customer segmentation project using unsupervised machine learning with MLOps practices including experiment tracking, data versioning, and deployment.

---

## About
Groups customers into distinct segments based on purchasing behavior and demographics using unsupervised machine learning.

---

## Dataset
| Property | Details |
|----------|---------|
| Source | Marketing Campaign dataset |
| Size | 2,240 rows × 29 columns |
| Format | Tab-separated CSV |
| Features | Demographics, spending, purchase channels, campaign responses |

---

## Approach

### 1. Preprocessing (`preprocessing.py`)
- Duplicates removed
- Education mapped → `Undergraduate`, `Graduate`, `Postgraduate`
- Marital status consolidated → `Single`, `Partnered`
- Income outliers (> 600K) removed
- Missing income rows dropped
- Irrelevant columns dropped (`ID`, `Response`, `Z_CostContact`, `Z_Revenue`, `Complain`)

### 2. Feature Engineering (`feature_engineering.py`)
- `Age` derived from `2026 − Year_Birth`
- Age outliers (> 100) removed
- Created: `Total_Children`, `Total_Campaign_Accepted`, `Customer_Tenure_Days`, `Total_Purchases`, `Total_Spending`, `Spending_Per_Purchase`
- Log-transform (`log1p`) applied to skewed spend columns
- One-hot encoding on `Marital_Status` and `Education_Group`

### 3. Scaling & Dimensionality Reduction
- `StandardScaler` fit on engineered features → saved as `scaler.pkl`
- PCA retaining 90% variance → saved as `pca.pkl`

### 4. Optimal K Selection (`clustering.py`)
- Elbow method (WCSS) evaluated across K = 2–10
- Silhouette scores evaluated across K = 2–10

### 5. Clustering (`clustering.py`)
- K-Means (K = 3, `n_init=10`) as final model
- Agglomerative Clustering used for comparison

### 6. Evaluation (`evaluation.py`)
- Davies–Bouldin Index for cluster quality

---

## Models Used
| Model | Type | Result |
|-------|------|--------|
| K-Means | Partition-based | ✅ Best |
| Agglomerative | Hierarchical | Comparison |

---

## Results

**Best Model:** K-Means with 3 clusters

| Cluster | Label | Profile |
|---------|-------|---------|
| 0 | 🟣 Budget Conscious Families | Low income, many children, price-sensitive |
| 1 | 🟡 High Value Loyalists | Highest income, max spending, campaign-responsive |
| 2 | 🔵 Middle Class Actives | Mid-range income, moderate spending across all channels |

**Top Features:** `Income`, `Total_Spending`, `Total_Purchases`, `Age`, `Total_Campaign_Accepted`

---

## MLflow Experiment Tracking

All experiments tracked using MLflow to compare clustering runs.

### Running MLflow
```bash
# Start MLflow dashboard
mlflow ui

# Run experiments
python src/log_experiment.py
```

### Experiments Conducted
| Run | n_clusters | init | max_iter | silhouette |
|-----|------------|------|----------|------------|
| kmeans-k2-k-means++-iter100 | 2 | k-means++ | 100 | 0.2859 |
| kmeans-k2-random-iter100 | 2 | random | 100 | 0.2859 |
| kmeans-k3-k-means++-iter100 | 3 | k-means++ | 100 | 0.1998 |
| kmeans-k4-random-iter100 | 4 | random | 100 | 0.2058 |
| kmeans-k5-random-iter100 | 5 | random | 100 | 0.1621 |

### Best Model (Auto-selected)
- **Mathematically best:** k=2, silhouette=0.2859
- **Business recommendation:** k=3, k-means++, iter=100
- **Saved to:** `models/best_kmeans_model.pkl`

---

## Project Structure
```
customer-segmentation-mlops/
├── data/
│   ├── raw/                         # Original dataset
│   └── processed/                   # Cleaned & clustered output CSV
├── models/                          # best_kmeans_model.pkl
├── notebooks/
│   └── project_analysis.ipynb       # EDA & exploration notebook
├── outputs/
│   ├── models/                      # scaler.pkl, pca.pkl, kmeans_model.pkl
│   ├── clusters/                    # Cluster assignment CSV
│   └── reports/                     # Plots & silhouette scores
├── reports/
│   └── experiment_report.md         # MLflow experiment findings
├── src/
│   ├── preprocessing.py             # Cleaning & encoding prep
│   ├── feature_engineering.py       # Feature creation & scaling
│   ├── clustering.py                # PCA, KMeans & Agglomerative
│   ├── evaluation.py                # Davies-Bouldin scoring
│   ├── pipeline.py                  # End-to-end pipeline
│   └── log_experiment.py            # MLflow experiment tracking
├── app.py                           # Streamlit predictor
├── main.py                          # Pipeline entry point
├── requirements.txt
└── .gitignore
```

---

## How to Run Locally
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models and save artifacts
python main.py

# 3. Track experiments with MLflow
mlflow ui
python src/log_experiment.py

# 4. Run the Streamlit app
python -m streamlit run app.py
```
