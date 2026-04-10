import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal

app = FastAPI(
    title="Customer Segmentation API",
    description="Predicts customer segment using KMeans clustering",
    version="1.0.0"
)

# ── Load models ──
scaler = joblib.load("models/scaler.pkl")
pca    = joblib.load("models/pca.pkl")
model  = joblib.load("models/kmeans_model.pkl")

print(f"[INFO] Loaded model: {type(model).__name__}")

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

CLUSTER_INFO = {
    0: "Budget Conscious Families",
    1: "High Value Loyalists",
    2: "Middle Class Actives"
}

# ── Pydantic Input Model ──
class CustomerInput(BaseModel):
    income:             float = Field(..., ge=0, le=200000)     #The ... means the field is required (not optional).
    recency:            int   = Field(..., ge=0, le=99)
    age:                int   = Field(..., ge=18, le=100)
    total_children:     int   = Field(..., ge=0, le=5)
    tenure_days:        int   = Field(..., ge=0, le=700)
    mnt_wines:          float = Field(..., ge=0, le=1500)
    mnt_fruits:         float = Field(..., ge=0, le=200)
    mnt_meat:           float = Field(..., ge=0, le=1500)
    mnt_fish:           float = Field(..., ge=0, le=300)
    mnt_sweet:          float = Field(..., ge=0, le=300)
    mnt_gold:           float = Field(..., ge=0, le=400)
    num_web_purchases:  int   = Field(..., ge=0, le=27)
    num_store_purchases:int   = Field(..., ge=0, le=20)
    num_catalog:        int   = Field(..., ge=0, le=30)
    num_deals:          int   = Field(..., ge=0, le=15)
    num_web_visits:     int   = Field(..., ge=0, le=20)
    campaigns_accepted: int   = Field(..., ge=0, le=6)
    marital_status:     Literal["Partnered", "Single", "Widow"]
    education_group:    Literal["Postgraduate", "Undergraduate", "Other"]

# ── Routes ──
@app.get("/")
def health():
    return {
        "status": "ok",
        "model": "KMeans Customer Segmentation"
        }

@app.post("/predict")
def predict(customer: CustomerInput):
    # ── Derived features ──
    total_spending        = (customer.mnt_wines + customer.mnt_fruits +
                             customer.mnt_meat  + customer.mnt_fish  +
                             customer.mnt_sweet + customer.mnt_gold)
    total_purchases       = (customer.num_web_purchases + customer.num_store_purchases +
                             customer.num_catalog + customer.num_deals)
    spending_per_purchase = total_spending / (total_purchases + 1)

    # ── One-hot encoding ──
    marital_partnered  = 1 if customer.marital_status == "Partnered" else 0
    marital_single     = 1 if customer.marital_status == "Single"    else 0
    marital_widow      = 1 if customer.marital_status == "Widow"     else 0
    education_postgrad = 1 if customer.education_group == "Postgraduate"  else 0
    education_undergrad= 1 if customer.education_group == "Undergraduate" else 0

    # ── Build DataFrame ──
    input_data = pd.DataFrame([[
        customer.income,
        customer.recency,
        np.log1p(customer.mnt_wines),  
        np.log1p(customer.mnt_fruits),
        np.log1p(customer.mnt_meat),   
        np.log1p(customer.mnt_fish),
        np.log1p(customer.mnt_sweet),  
        np.log1p(customer.mnt_gold),
        customer.num_deals, 
        customer.num_web_purchases,
        customer.num_catalog, 
        customer.num_store_purchases,
        customer.num_web_visits, 
        customer.age,
        customer.total_children, 
        customer.campaigns_accepted,
        customer.tenure_days, 
        total_purchases,
        total_spending, 
        spending_per_purchase,
        marital_partnered, 
        marital_single, 
        marital_widow,
        education_postgrad, 
        education_undergrad
    ]], columns=FEATURE_COLUMNS)

    # ── Pipeline: scale → PCA → predict ──
    input_scaled    = scaler.transform(input_data)
    input_scaled_df = pd.DataFrame(input_scaled, columns=FEATURE_COLUMNS)
    input_pca       = pca.transform(input_scaled_df)
    cluster         = int(model.predict(input_pca)[0])

    return {
        "cluster_id":   cluster,
        "segment_name": CLUSTER_INFO[cluster],
        "derived": {
            "total_spending":        round(total_spending, 2),
            "total_purchases":       total_purchases,
            "spending_per_purchase": round(spending_per_purchase, 2)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=False)


