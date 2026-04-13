import pandas as pd
import numpy as np
from scipy import stats

FEATURE_COLUMNS = [
    'Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts',
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
    'NumStorePurchases', 'NumWebVisitsMonth', 'Age', 'Total_Children',
    'Total_Campaign_Accepted', 'Customer_Tenure_Days', 'Total_Purchases',
    'Total_Spending', 'Spending_Per_Purchase',
    'Marital_Status_Partnered', 'Marital_Status_Single',
    'Marital_Status_Widow', 'Education_Group_Postgraduate',
    'Education_Group_Undergraduate'
]

print("Loading reference data...")
reference = pd.read_csv("data/marketing_data_cleaned.csv")[FEATURE_COLUMNS]

print("Simulating current data...")
current = reference.copy()
current['Income'] = current['Income'] * 1.3
current['Age'] = current['Age'] + 5
current['MntWines'] = current['MntWines'] * 0.7

print("\n" + "="*60)
print("DRIFT REPORT")
print("="*60)

drifted = []

for col in FEATURE_COLUMNS:
    # KS test: checks if two distributions are significantly different
    ks_stat, p_value = stats.ks_2samp(reference[col], current[col])
    is_drifted = p_value < 0.05  # standard threshold

    status = "⚠️  DRIFTED" if is_drifted else "✅ OK"
    print(f"{status} | {col:<35} | p-value: {p_value:.4f}")

    if is_drifted:
        drifted.append(col)

print("="*60)
print(f"\nTotal drifted: {len(drifted)}/{len(FEATURE_COLUMNS)} columns")
print(f"Drifted columns: {drifted}")

import joblib

print("\n" + "="*60)
print("PREDICTION DISTRIBUTION")
print("="*60)

# Load models
scaler = joblib.load("models/scaler.pkl")
pca    = joblib.load("models/pca.pkl")
model  = joblib.load("models/kmeans_model.pkl")

CLUSTER_NAMES = {
    0: "Budget Conscious Families",
    1: "High Value Loyalists",
    2: "Middle Class Actives"
}

for label, data in [("Reference", reference), ("Current", current)]:
    X_scaled = scaler.transform(data)
    X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS)
    X_pca = pca.transform(X_scaled_df)
    predictions = model.predict(X_pca)

    print(f"\n{label} Data:")
    counts = pd.Series(predictions).value_counts().sort_index()
    for cluster_id, count in counts.items():
        pct = count / len(predictions) * 100
        print(f"  Cluster {cluster_id} ({CLUSTER_NAMES[cluster_id]}): {count} ({pct:.1f}%)")

