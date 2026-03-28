import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the full pipeline
scaler = joblib.load("models/scaler.pkl")
pca    = joblib.load("models/pca.pkl")
model  = joblib.load("models/kmeans_model.pkl")

# This will print the ACTUAL model class name — confirms what was loaded
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


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": "KMeans Customer Segmentation"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if data is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    required = [
        "income", "recency", "age", "total_children", "tenure_days",
        "mnt_wines", "mnt_fruits", "mnt_meat", "mnt_fish", "mnt_sweet", "mnt_gold",
        "num_web_purchases", "num_store_purchases", "num_catalog", "num_deals",
        "num_web_visits", "campaigns_accepted"
    ]
    for key in required:
        if key not in data:
            return jsonify({"error": f"Missing field: '{key}'"}), 400

    # ── Extract raw values ──
    mnt_wines   = data["mnt_wines"]
    mnt_fruits  = data["mnt_fruits"]
    mnt_meat    = data["mnt_meat"]
    mnt_fish    = data["mnt_fish"]
    mnt_sweet   = data["mnt_sweet"]
    mnt_gold    = data["mnt_gold"]
    num_web     = data["num_web_purchases"]
    num_store   = data["num_store_purchases"]
    num_catalog = data["num_catalog"]
    num_deals   = data["num_deals"]

    # ── Derived features ──
    total_spending        = mnt_wines + mnt_fruits + mnt_meat + mnt_fish + mnt_sweet + mnt_gold
    total_purchases       = num_web + num_store + num_catalog + num_deals
    spending_per_purchase = total_spending / (total_purchases + 1)

    # ── Build input DataFrame with log transforms on spending columns ──
    input_data = pd.DataFrame([[
        data["income"],
        data["recency"],
        np.log1p(mnt_wines),
        np.log1p(mnt_fruits),
        np.log1p(mnt_meat),
        np.log1p(mnt_fish),
        np.log1p(mnt_sweet),
        np.log1p(mnt_gold),
        num_deals,
        num_web,
        num_catalog,
        num_store,
        data["num_web_visits"],
        data["age"],
        data["total_children"],
        data["campaigns_accepted"],
        data["tenure_days"],
        total_purchases,
        total_spending,
        spending_per_purchase,
        1, 0, 0, 0, 0
    ]], columns=FEATURE_COLUMNS)

    # ── Pipeline: scale → PCA → predict ──
    input_scaled = scaler.transform(input_data)
    input_scaled_df = pd.DataFrame(input_scaled, columns=FEATURE_COLUMNS)
    input_pca    = pca.transform(input_scaled_df)
    cluster      = int(model.predict(input_pca)[0])

    return jsonify({
        "cluster_id":   cluster,
        "segment_name": CLUSTER_INFO[cluster],
        "derived": {
            "total_spending":        round(total_spending, 2),
            "total_purchases":       total_purchases,
            "spending_per_purchase": round(spending_per_purchase, 2)
        }
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)