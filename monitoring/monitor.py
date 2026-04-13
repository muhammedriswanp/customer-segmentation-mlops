import pandas as pd
import os
from evidently import Report
from evidently.presets import DataDriftPreset

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
reference_data = pd.read_csv("data/marketing_data_cleaned.csv")[FEATURE_COLUMNS]

print("Simulating current data...")
current_data = reference_data.copy()
current_data['Income'] = current_data['Income'] * 1.3
current_data['Age'] = current_data['Age'] + 5
current_data['MntWines'] = current_data['MntWines'] * 0.7

print("Generating drift report...")
report = Report(metrics=[DataDriftPreset()])
snapshot = report.run(reference_data=reference_data, current_data=current_data)

os.makedirs("reports/monitoring", exist_ok=True)
snapshot.save_html("reports/monitoring/drift_report.html")
print("✅ Report saved to reports/monitoring/drift_report.html")

# start reports/monitoring/drift_report.html