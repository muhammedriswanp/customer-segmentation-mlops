import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the full pipeline
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

def validate_input(data):
    # ── Level 1: Key validation  ──
    required = [
        "income", "recency", "age", "total_children", "tenure_days",
        "mnt_wines", "mnt_fruits", "mnt_meat", "mnt_fish", "mnt_sweet", "mnt_gold",
        "num_web_purchases", "num_store_purchases", "num_catalog", "num_deals",
        "num_web_visits", "campaigns_accepted",
        "marital_status", "education_group"  # ← ADDED
    ]

    for key in required:
        if key not in data:
            return f"Missing field: '{key}'"
        
    # ── Level 2: Type validation ──
    # Numeric fields
    numeric_fields = [
        "income", "recency", "age", "total_children", "tenure_days",
        "mnt_wines", "mnt_fruits", "mnt_meat", "mnt_fish", "mnt_sweet", "mnt_gold",
        "num_web_purchases", "num_store_purchases", "num_catalog", "num_deals",
        "num_web_visits", "campaigns_accepted"
    ]

    for field in numeric_fields:
        if not isinstance(data[field], (int, float)):
            return f"'{field}' must be a number, got {type(data[field]).__name__}"

    # Categorical fields
    categorical_fields = ["marital_status", "education_group"]
    for field in categorical_fields:
        if not isinstance(data[field], str):
            return f"'{field}' must be a string, got {type(data[field]).__name__}"

    # ── Level 2.5: Integer check for whole-number fields ──
    whole_number_fields = [
        "total_children", "campaigns_accepted",
        "num_web_purchases", "num_store_purchases",
        "num_catalog", "num_deals", "num_web_visits",
        "recency", "age", "tenure_days"
    ]

    for field in whole_number_fields:
        if isinstance(data[field], float) and not data[field].is_integer():
            return f"'{field}' must be a whole number, got {data[field]}"

    # ── Level 3: Range validation ──
    ranges = {
        "income":              (0,    200000),  
        "age":                 (18,   100),
        "recency":             (0,    99),      
        "total_children":      (0,    5),
        "tenure_days":         (0,    700),     
        "mnt_wines":           (0,    1500),
        "mnt_fruits":          (0,    200),
        "mnt_meat":            (0,    1500),
        "mnt_fish":            (0,    300),
        "mnt_sweet":           (0,    300),
        "mnt_gold":            (0,    400),
        "num_web_purchases":   (0,    27),     
        "num_store_purchases": (0,    20),
        "num_catalog":         (0,    30),
        "num_deals":           (0,    15),
        "num_web_visits":      (0,    20),
        "campaigns_accepted":  (0,    6),
    }

    for field, (min_val, max_val) in ranges.items():
        value = data[field]
        if value < min_val or value > max_val:
            return f"'{field}' must be between {min_val} and {max_val}, got {value}"

    # ── Level 3.5: Categorical validation ──
    VALID_MARITAL_STATUS = ["Partnered", "Single", "Widow"]
    VALID_EDUCATION_GROUP = ["Postgraduate", "Undergraduate", "Other"]
    
    if data["marital_status"] not in VALID_MARITAL_STATUS:
        return f"'marital_status' must be one of {VALID_MARITAL_STATUS}, got '{data['marital_status']}'"
    
    if data["education_group"] not in VALID_EDUCATION_GROUP:
        return f"'education_group' must be one of {VALID_EDUCATION_GROUP}, got '{data['education_group']}'"
        
    return None  # ← None means no error, everything is valid


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
    
    # ── Run all validations ──
    error = validate_input(data)
    if error:
        return jsonify({"error": error}), 400


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

    # ── Dynamic one-hot encoding for categoricals ──
    marital_status = data["marital_status"]
    education_group = data["education_group"]
    
    marital_partnered = 1 if marital_status == "Partnered" else 0
    marital_single = 1 if marital_status == "Single" else 0
    marital_widow = 1 if marital_status == "Widow" else 0
    education_postgrad = 1 if education_group == "Postgraduate" else 0
    education_undergrad = 1 if education_group == "Undergraduate" else 0

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
        marital_partnered, marital_single, marital_widow,
        education_postgrad, education_undergrad  
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
    