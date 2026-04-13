import pandas as pd
import joblib
import os
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

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

# ── Load data ──
print("Loading reference data...")
reference_data = pd.read_csv("data/marketing_data_cleaned.csv")[FEATURE_COLUMNS]

print("Simulating current data...")
current_data = reference_data.copy()
current_data['Income'] = current_data['Income'] * 1.3
current_data['Age'] = current_data['Age'] + 5
current_data['MntWines'] = current_data['MntWines'] * 0.7

# ── Load models ──
scaler = joblib.load("models/scaler.pkl")
pca    = joblib.load("models/pca.pkl")
model  = joblib.load("models/kmeans_model.pkl")

# ── Add predictions as a column ──
def get_predictions(df):
    X_scaled = scaler.transform(df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS)
    X_pca = pca.transform(X_scaled_df)
    return model.predict(X_pca)

reference_data['Cluster'] = get_predictions(reference_data)
current_data['Cluster']   = get_predictions(current_data)

# ── Generate report ──
print("Generating drift report...")
report = Report(metrics=[
    DataDriftPreset(),      # checks input feature drift,   Has the data distribution changed?
    DataSummaryPreset()     # summarizes prediction column distribution, What exactly does the data look like now vs before
])
snapshot = report.run(
    reference_data=reference_data,
    current_data=current_data,
)

os.makedirs("reports/monitoring", exist_ok=True)
snapshot.save_html("reports/monitoring/drift_report.html")
print("✅ Report saved to reports/monitoring/drift_report.html")

# start reports/monitoring/drift_report.html
# py -3.11 → runs Python 3.11 → evidently works ✅
