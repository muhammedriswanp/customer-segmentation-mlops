# Customer Segmentation & Clustering

Groups customers into distinct segments based on purchasing behavior and demographics using unsupervised machine learning.

## Dataset

**Source:** Marketing Campaign dataset (`data/raw/marketing_campaign.csv`)  
**Size:** 2,240 rows × 29 columns (tab-separated)  
**Features:** Demographics, spending amounts, purchase channels, and campaign responses

## Approach

- **Preprocessing** — duplicates removed, education mapped (`Basic`/`2n Cycle` → `Undergraduate`, `Master`/`PhD` → `Postgraduate`), marital status consolidated to `Single`/`Partnered`, income outliers (> 600K) removed, missing income dropped, irrelevant columns (`ID`, `Response`, `Z_CostContact`, `Z_Revenue`, `Complain`) dropped (`preprocessing.py`)
- **Feature Engineering** — `Age` derived from 2026 − `Year_Birth`; age outliers (> 100) removed; `Total_Children`, `Total_Campaign_Accepted`, `Customer_Tenure_Days`, `Total_Purchases`, `Total_Spending`, `Spending_Per_Purchase` created; log-transform (`log1p`) applied to skewed spend columns; one-hot encoding on `Marital_Status` and `Education_Group` (`feature_engineering.py`)
- **Scaling** — `StandardScaler` fit on engineered features, saved as `scaler.pkl` (`pipeline.py`)
- **Dimensionality Reduction** — PCA retaining 90% variance, saved as `pca.pkl` (`clustering.py`)
- **Optimal K Selection** — Elbow method (WCSS) + Silhouette scores evaluated across K = 2–10 (`clustering.py`)
- **Clustering** — K-Means (K = 3, `n_init=10`) as final model; Agglomerative Clustering used for comparison (`clustering.py`)
- **Evaluation** — Davies–Bouldin Index for cluster quality (`evaluation.py`)

## Models Used

| Model | Type |
|---|---|
| K-Means | Partition-based ✅ Best |
| Agglomerative | Hierarchical (comparison) |

## Results

**Best Model:** K-Means with **3 clusters**

| Cluster | Label | Profile |
|---|---|---|
| 0 | 🟣 Budget Conscious Families | Low income, many children, price-sensitive, frequent web visits |
| 1 | 🟡 High Value Loyalists | Highest income, max spending, campaign-responsive, prefer catalogues |
| 2 | 🔵 Middle Class Actives | Mid-range income, loyal, moderate spending across all channels |

Top features: `Income`, `Total_Spending`, `Total_Purchases`, `Age`, `Total_Campaign_Accepted`

## Project Structure

```
customer-segmentation-clustering/
├── data/
│   ├── raw/                         # Original dataset
│   └── processed/                   # Cleaned & clustered output CSV
├── notebooks/
│   └── project_analysis.ipynb       # EDA & exploration notebook
├── outputs/
│   ├── models/                      # scaler.pkl, pca.pkl, kmeans_model.pkl
│   ├── clusters/                    # Cluster assignment CSV
│   └── reports/                     # Plots & reports (e.g. silhouette_scores.png)
├── src/
│   ├── preprocessing.py             # Cleaning, outlier removal & encoding prep
│   ├── feature_engineering.py       # Feature creation, log-transform, encoding & scaling
│   ├── clustering.py                # PCA, Elbow, Silhouette, KMeans & Agglomerative helpers
│   ├── evaluation.py                # Davies–Bouldin scoring
│   └── pipeline.py                  # End-to-end pipeline — trains & saves models
├── app.py                           # Streamlit customer segment predictor
├── main.py                          # Entry point to run the pipeline
└── requirements.txt
```

## How to Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models and save artefacts (scaler, PCA, KMeans)
python main.py

# 3. Run the Streamlit app
python -m streamlit run app.py
```
